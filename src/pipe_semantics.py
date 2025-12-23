from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeSemanticsMixin:
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string using the configured embedding model with caching.

        The embedding model is expected to be served behind an OpenAI-compatible
        `/v1/embeddings` endpoint.
        """
        if not text or not text.strip():
            return None

        text = text[:2000]
        text = text.replace(":", " - ")

        # Check cache first
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding

        # If not in cache, get from API
        try:
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": text,
                }

                async with session.post(
                    f"{self.valves.EMBEDDING_API_BASE}/v1/embeddings",
                    json=payload,
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding: Optional[List[float]] = None

                        # OpenAI-style response: {"data": [{"embedding": [...]}], ...}
                        if isinstance(result, dict) and "data" in result:
                            data = result.get("data") or []
                            if isinstance(data, list) and data:
                                first_item = data[0] or {}
                                embedding = first_item.get("embedding")

                        # Fallback: some backends might return {"embedding": [...]} directly
                        if embedding is None and "embedding" in result:
                            embedding = result.get("embedding")

                        if embedding:
                            # Cache the result
                            self.embedding_cache.set(text, embedding)

                            # Store embedding dimension the first time we see it
                            if self.embedding_dim is None:
                                try:
                                    self.embedding_dim = len(embedding)
                                    logger.info(
                                        f"Detected embedding dimension: {self.embedding_dim}"
                                    )
                                except Exception:
                                    pass

                            return embedding
                    else:
                        logger.warning(
                            f"Embedding request failed with status {response.status}"
                        )

            return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    async def get_transformed_embedding(
        self, text: str, transformation=None
    ) -> Optional[List[float]]:
        """Get embedding with optional transformation applied, using caching for efficiency"""
        if not text or not text.strip():
            return None

        # If no transformation needed, just return regular embedding
        if transformation is None:
            return await self.get_embedding(text)

        # Check transformation cache first - simple lookup
        transform_id = (
            transformation.get("id", str(hash(str(transformation))))
            if isinstance(transformation, dict)
            else transformation
        )
        cached_transformed = self.transformation_cache.get(text, transform_id)
        if cached_transformed is not None:
            return cached_transformed

        # If not in transformation cache, get base embedding
        base_embedding = await self.get_embedding(text)
        if not base_embedding:
            return None

        # Apply transformation
        transformed = await self.apply_semantic_transformation(
            base_embedding, transformation
        )

        # Cache the transformed result only if successful
        if transformed:
            self.transformation_cache.set(text, transform_id, transformed)

        return transformed

    async def compute_semantic_eigendecomposition(
        self, chunks, embeddings, cache_key=None
    ):
        """Perform semantic eigendecomposition on chunk embeddings with caching"""
        if not chunks or not embeddings or len(chunks) < 3:
            return None

        # Generate cache key if not provided
        if cache_key is None:
            # Create a stable cache key based on embeddings fingerprint
            embeddings_concat = np.concatenate(
                embeddings[: min(5, len(embeddings))], axis=0
            )
            fingerprint = np.mean(embeddings_concat, axis=0)
            cache_key = hash(str(fingerprint.round(2)))

        # Check cache first
        state = self.get_state()
        eigendecomposition_cache = state.get("eigendecomposition_cache", {})
        if cache_key in eigendecomposition_cache:
            return eigendecomposition_cache[cache_key]

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)

            # Check for invalid values
            if np.isnan(embeddings_array).any() or np.isinf(embeddings_array).any():
                logger.warning(
                    "Invalid values in embeddings, cannot perform eigendecomposition"
                )
                return None

            # Center the embeddings
            centered_embeddings = embeddings_array - np.mean(embeddings_array, axis=0)

            # Compute covariance matrix
            cov_matrix = np.cov(centered_embeddings, rowvar=False)

            # Perform eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Determine how many principal components to keep
            total_variance = np.sum(eigenvalues)
            if total_variance <= 0:
                logger.warning(
                    "Total variance is zero or negative, cannot continue with eigendecomposition"
                )
                return None

            explained_variance_ratio = eigenvalues / total_variance

            # Keep components that explain 80% of variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumulative_variance >= 0.8) + 1

            # Cap to available dimensionality (no artificial minimum that can desync shapes)
            available_dims = min(len(eigenvalues), embeddings_array.shape[1])
            n_components = max(1, min(n_components, available_dims, 10))

            # Project embeddings onto principal components
            principal_components = eigenvectors[:, :n_components]
            projected_embeddings = np.dot(centered_embeddings, principal_components)

            result = {
                "eigenvalues": eigenvalues[:n_components].tolist(),
                "eigenvectors": principal_components.tolist(),
                "explained_variance": explained_variance_ratio[:n_components].tolist(),
                "projected_embeddings": projected_embeddings.tolist(),
                "n_components": n_components,
            }

            # Cache the result
            eigendecomposition_cache[cache_key] = result
            # Limit cache size
            if (
                len(eigendecomposition_cache) > 50
            ):  # Store up to 50 different decompositions
                oldest_key = next(iter(eigendecomposition_cache))
                del eigendecomposition_cache[oldest_key]
            self.update_state("eigendecomposition_cache", eigendecomposition_cache)

            return result
        except Exception as e:
            logger.error(f"Error in semantic eigendecomposition: {e}")
            return None

    async def create_semantic_transformation(
        self, semantic_eigendecomposition, pdv=None, trajectory=None, gap_vector=None
    ):
        """Create a semantic transformation matrix based on eigendecomposition and direction vectors"""
        if not semantic_eigendecomposition:
            return None

        # Generate a unique ID for this transformation
        state = self.get_state()
        transformation_id = f"transform_{hash(str(pdv))[:8]}_{hash(str(trajectory))[:8]}_{hash(str(gap_vector))[:8]}"

        try:
            # Get principal components
            eigenvectors = np.array(semantic_eigendecomposition["eigenvectors"])
            eigenvalues = np.array(semantic_eigendecomposition["eigenvalues"])

            # Create initial transformation (identity)
            embedding_dim = eigenvectors.shape[0]
            transformation = np.eye(embedding_dim)

            # Get importance weights for each eigenvector
            variance_importance = eigenvalues / np.sum(eigenvalues)

            # Enhance dimensions based on eigenvalues (semantic importance)
            for i, importance in enumerate(variance_importance):
                eigenvector = eigenvectors[:, i]
                # Scale amplification by dimension importance
                amplification = 1.0 + importance * 2.0  # 1.0 to 3.0
                # Add outer product to emphasize this dimension
                transformation += (amplification - 1.0) * np.outer(
                    eigenvector, eigenvector
                )

            # Calculate weights for different direction vectors
            pdv_weight = (
                self.valves.SEMANTIC_TRANSFORMATION_STRENGTH
                * state.get("user_preferences", {}).get("impact", 0.0)
                if pdv is not None
                else 0.0
            )

            # Calculate trajectory weight
            trajectory_weight = (
                self.valves.TRAJECTORY_MOMENTUM if trajectory is not None else 0.0
            )

            # Calculate adaptive gap weight based on research progress
            gap_weight = 0.0
            if gap_vector is not None:
                # Get current cycle and max cycles for adaptive calculation
                current_cycle = len(state.get("cycle_summaries", [])) + 1
                max_cycles = self.valves.MAX_CYCLES
                fade_start_cycle = min(5, int(0.5 * max_cycles))

                # Get gap coverage history to analyze trend
                gap_coverage_history = state.get("gap_coverage_history", [])

                # Determine if gaps are still valuable for research direction
                if current_cycle <= fade_start_cycle:
                    # Early cycles: use full gap weight
                    gap_weight = self.valves.GAP_EXPLORATION_WEIGHT
                else:
                    # Calculate adaptive weight based on research progress
                    # Linear fade from full weight to zero
                    remaining_cycles = max_cycles - current_cycle
                    total_fade_cycles = max_cycles - fade_start_cycle
                    if total_fade_cycles > 0:  # Avoid division by zero
                        fade_ratio = remaining_cycles / total_fade_cycles
                        gap_weight = self.valves.GAP_EXPLORATION_WEIGHT * max(
                            0.0, fade_ratio
                        )
                    else:
                        gap_weight = 0.0

            # Normalize weights to sum to at most 0.8 (leaving some room for the eigendecomposition base)
            total_weight = pdv_weight + trajectory_weight + gap_weight
            if total_weight > 0.8:
                scale_factor = 0.8 / total_weight
                pdv_weight *= scale_factor
                trajectory_weight *= scale_factor
                gap_weight *= scale_factor

            # Apply PDV transformation
            if pdv is not None and pdv_weight > 0.1:
                pdv_array = np.array(pdv)
                norm = np.linalg.norm(pdv_array)
                if norm > 1e-10:
                    pdv_array = pdv_array / norm
                    transformation += pdv_weight * np.outer(pdv_array, pdv_array)

            # Apply trajectory transformation
            if trajectory is not None and trajectory_weight > 0.1:
                trajectory_array = np.array(trajectory)
                norm = np.linalg.norm(trajectory_array)
                if norm > 1e-10:
                    trajectory_array = trajectory_array / norm
                    transformation += trajectory_weight * np.outer(
                        trajectory_array, trajectory_array
                    )

            # Apply gap vector transformation
            if gap_vector is not None and gap_weight > 0.1:
                gap_array = np.array(gap_vector)
                norm = np.linalg.norm(gap_array)
                if norm > 1e-10:
                    gap_array = gap_array / norm
                    transformation += gap_weight * np.outer(gap_array, gap_array)

            return {
                "id": transformation_id,
                "matrix": transformation.tolist(),
                "dimension": embedding_dim,
                "pdv_weight": pdv_weight,
                "trajectory_weight": trajectory_weight,
                "gap_weight": gap_weight,
            }

        except Exception as e:
            logger.error(f"Error creating semantic transformation: {e}")
            return None

    async def apply_semantic_transformation(self, embedding, transformation):
        """Apply semantic transformation to an embedding"""
        if not transformation or not embedding:
            return embedding

        try:
            # Convert to numpy arrays
            embedding_array = np.array(embedding)

            # If transformation is an ID string, look up the transformation
            if isinstance(transformation, str):
                # In a real implementation, retrieve from cache/storage
                logger.warning(f"Transformation ID not found: {transformation}")
                return embedding

            # Guard against unexpected transformation payloads
            if not isinstance(transformation, dict) or "matrix" not in transformation:
                logger.warning("Invalid transformation payload; skipping transformation")
                return embedding

            # If it's a transformation object, get the matrix
            transform_matrix = np.array(transformation["matrix"])

            # Check for invalid values
            if (
                np.isnan(embedding_array).any()
                or np.isnan(transform_matrix).any()
                or np.isinf(embedding_array).any()
                or np.isinf(transform_matrix).any()
            ):
                logger.warning("Invalid values in embedding or transformation matrix")
                return embedding

            # Apply transformation
            transformed = np.dot(embedding_array, transform_matrix)

            # Check for valid result
            if np.isnan(transformed).any() or np.isinf(transformed).any():
                logger.warning("Transformation produced invalid values")
                return embedding

            # Normalize to unit vector
            norm = np.linalg.norm(transformed)
            if norm > 1e-10:  # Avoid division by near-zero
                transformed = transformed / norm
                return transformed.tolist()
            else:
                logger.warning("Transformation produced zero vector")
                return embedding
        except Exception as e:
            logger.error(f"Error applying semantic transformation: {e}")
            return embedding

    async def calculate_research_trajectory(self, previous_queries, successful_results):
        """Calculate the research trajectory based on successful searches from recent cycles only"""
        if not previous_queries or not successful_results:
            return None

        # Check trajectory cache to avoid expensive recalculation
        state = self.get_state()
        trajectory_cache = state.get("trajectory_cache", {})

        # Use limited recent items to create cache key
        recent_query_key = hash(
            str(
                previous_queries[-3:]
                if len(previous_queries) >= 3
                else previous_queries
            )
        )
        recent_result_key = hash(
            str([r.get("url", "") for r in successful_results[-5:] if "url" in r])
        )
        cache_key = f"{recent_query_key}_{recent_result_key}"

        if cache_key in trajectory_cache:
            logger.info(f"Using cached trajectory for key: {cache_key}")
            return trajectory_cache[cache_key]

        # Use the trajectory accumulator if initialized
        if self.trajectory_accumulator is None:
            # Initialize with first sample embedding dimension
            sample_embedding = None
            for result in successful_results[:6]:
                content = result.get("content", "")[:2000]
                if content:
                    sample_embedding = await self.get_embedding(content)
                    if sample_embedding:
                        embedding_dim = len(sample_embedding)
                        self.trajectory_accumulator = TrajectoryAccumulator(
                            embedding_dim
                        )
                        break

            # If we couldn't get a sample, use default dimension
            if not sample_embedding:
                fallback_dim = self.embedding_dim or 384
                self.trajectory_accumulator = TrajectoryAccumulator(fallback_dim)

        try:
            # Limit to last 5 cycles worth of data for efficiency
            max_history_cycles = 5
            queries_per_cycle = self.valves.SEARCH_RESULTS_PER_QUERY
            results_per_query = self.valves.SUCCESSFUL_RESULTS_PER_QUERY

            # Calculate maximum items to keep
            max_queries = max_history_cycles * queries_per_cycle
            max_results = max_queries * results_per_query

            # Take only the most recent queries and results
            recent_queries = (
                previous_queries[-max_queries:]
                if len(previous_queries) > max_queries
                else previous_queries
            )
            recent_results = (
                successful_results[-max_results:]
                if len(successful_results) > max_results
                else successful_results
            )

            logger.info(
                f"Calculating research trajectory with {len(recent_queries)} recent queries and {len(recent_results)} recent results"
            )

            # Get embeddings for queries sequentially
            query_embeddings = []
            for query in recent_queries:
                embedding = await self.get_embedding(query)
                if embedding:
                    query_embeddings.append(embedding)

            # Process results sequentially
            result_embeddings = []
            for result in recent_results:
                content = result.get("content", "")
                if not content:
                    continue
                embedding = await self.get_embedding(content[:2000])
                if embedding:
                    result_embeddings.append(embedding)

            if not query_embeddings or not result_embeddings:
                return None

            # Update trajectory accumulator with new cycle data
            self.trajectory_accumulator.add_cycle_data(
                query_embeddings, result_embeddings
            )

            # Get accumulated trajectory
            trajectory = self.trajectory_accumulator.get_trajectory()

            # If trajectory exists and we have PDV, calculate alignment to track for adaptive fade-out
            if trajectory:
                # Store the trajectory
                trajectory_cache[cache_key] = trajectory
                # Limit cache size
                if len(trajectory_cache) > 10:
                    oldest_key = next(iter(trajectory_cache))
                    del trajectory_cache[oldest_key]
                self.update_state("trajectory_cache", trajectory_cache)

                # Calculate PDV alignment if available
                pdv = state.get("user_preferences", {}).get("pdv")
                if pdv:
                    # Calculate alignment between trajectory and PDV
                    pdv_array = np.array(pdv)
                    trajectory_array = np.array(trajectory)
                    alignment = np.dot(trajectory_array, pdv_array)
                    # Normalize to 0-1 range
                    alignment = (alignment + 1) / 2

                    # Store in alignment history
                    pdv_alignment_history = state.get("pdv_alignment_history", [])
                    pdv_alignment_history.append(alignment)
                    # Keep only recent history
                    if len(pdv_alignment_history) > 5:
                        pdv_alignment_history = pdv_alignment_history[-5:]
                    self.update_state("pdv_alignment_history", pdv_alignment_history)

                    logger.info(f"PDV-Trajectory alignment: {alignment:.3f}")

            return trajectory

        except Exception as e:
            logger.error(f"Error calculating research trajectory: {e}")
            return None

    async def calculate_gap_vector(self):
        """Calculate a vector pointing toward research gaps"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return None

        try:
            coverage = np.array(research_dimensions["coverage"])
            components = np.array(research_dimensions["eigenvectors"])

            # Get current cycle for adaptive calculations
            current_cycle = len(state.get("cycle_summaries", [])) + 1
            max_cycles = self.valves.MAX_CYCLES
            fade_start_cycle = min(5, int(0.5 * max_cycles))

            # Determine adaptive fade-out based on research progress
            fade_factor = 1.0
            if current_cycle > fade_start_cycle:
                # Linear fade from full influence to zero
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_factor = max(0.0, remaining_cycles / total_fade_cycles)
                else:
                    fade_factor = 0.0

            # Early exit if we've faded out completely
            if fade_factor <= 0.01:
                logger.info("Gap vector faded out completely, returning None")
                return None

            # Store gap coverage for tracking
            gap_coverage_history = state.get("gap_coverage_history", [])
            gap_coverage_history.append(np.mean(coverage).item())
            if len(gap_coverage_history) > 5:
                gap_coverage_history = gap_coverage_history[-5:]
            self.update_state("gap_coverage_history", gap_coverage_history)

            # Create a weighted sum of components based on coverage gaps
            gap_vector = np.zeros(components.shape[1])

            for i, cov in enumerate(coverage):
                # Calculate gap (1.0 - coverage)
                gap = 1.0 - cov

                # Only consider significant gaps
                if gap > 0.3:
                    # Ensure components is a numpy array
                    if isinstance(components, np.ndarray) and i < len(components):
                        # Weight by gap size - larger gaps have more influence
                        gap_vector += gap * components[i]
                    else:
                        logger.warning(f"Invalid components at index {i}")

            # Apply adaptive fade-out
            gap_vector *= fade_factor

            # Check for NaN or Inf values
            if np.isnan(gap_vector).any() or np.isinf(gap_vector).any():
                logger.warning("Invalid values in gap vector")
                return None

            # Normalize
            norm = np.linalg.norm(gap_vector)
            if norm > 1e-10:
                gap_vector = gap_vector / norm
                return gap_vector.tolist()
            else:
                logger.warning("Gap vector has zero norm")
                return None
        except Exception as e:
            logger.error(f"Error calculating gap vector: {e}")
            return None

    async def calculate_query_similarity(
        self,
        content_embedding: List[float],
        query_embedding: List[float],
        outline_embedding: Optional[List[float]] = None,
        summary_embedding: Optional[List[float]] = None,
    ) -> float:
        """Calculate similarity to query with optional context embeddings using caching"""

        # Get similarity cache
        state = self.get_state()
        similarity_cache = state.get("similarity_cache", {})

        # Generate cache keys for each embedding
        content_key = hash(str(np.array(content_embedding).round(2)))
        query_key = hash(str(np.array(query_embedding).round(2)))

        # First check if we have the full combined similarity cached
        combined_key = f"combined_{content_key}_{query_key}"
        if outline_embedding:
            outline_key = hash(str(np.array(outline_embedding).round(2)))
            combined_key += f"_{outline_key}"
        if summary_embedding:
            summary_key = hash(str(np.array(summary_embedding).round(2)))
            combined_key += f"_{summary_key}"

        if combined_key in similarity_cache:
            return similarity_cache[combined_key]

        # Convert to numpy arrays
        c_emb = np.array(content_embedding)
        q_emb = np.array(query_embedding)

        # Normalize embeddings
        c_emb = c_emb / np.linalg.norm(c_emb)
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Check cache for base query similarity
        base_key = f"{content_key}_{query_key}"
        if base_key in similarity_cache:
            query_sim = similarity_cache[base_key]
        else:
            # Base query similarity using cosine similarity
            query_sim = np.dot(c_emb, q_emb)
            # Cache the result
            similarity_cache[base_key] = query_sim

        # Weight factors
        query_weight = 0.4  # Primary query importance
        outline_weight = 0.3  # Research outline importance
        summary_weight = 0.3  # Previous summary importance

        # If we have an outline embedding, include it
        outline_sim = 0.0
        if outline_embedding is not None:
            # Check cache for outline similarity
            outline_key = hash(str(np.array(outline_embedding).round(2)))
            outline_cache_key = f"{content_key}_{outline_key}"

            if outline_cache_key in similarity_cache:
                outline_sim = similarity_cache[outline_cache_key]
            else:
                o_emb = np.array(outline_embedding)
                o_emb = o_emb / np.linalg.norm(o_emb)
                outline_sim = np.dot(c_emb, o_emb)
                # Cache the result
                similarity_cache[outline_cache_key] = outline_sim
        else:
            # Redistribute weight
            query_weight += outline_weight
            outline_weight = 0.0

        # If we have a summary embedding (for follow-ups), include it
        summary_sim = 0.0
        if summary_embedding is not None:
            # Check cache for summary similarity
            summary_key = hash(str(np.array(summary_embedding).round(2)))
            summary_cache_key = f"{content_key}_{summary_key}"

            if summary_cache_key in similarity_cache:
                summary_sim = similarity_cache[summary_cache_key]
            else:
                s_emb = np.array(summary_embedding)
                s_emb = s_emb / np.linalg.norm(s_emb)
                summary_sim = np.dot(c_emb, s_emb)
                # Cache the result
                similarity_cache[summary_cache_key] = summary_sim
        else:
            # Redistribute weight
            query_weight += summary_weight
            summary_weight = 0.0

        # Weighted combination of similarities
        combined_sim = (
            (query_sim * query_weight)
            + (outline_sim * outline_weight)
            + (summary_sim * summary_weight)
        )

        # Cache the combined result
        similarity_cache[combined_key] = combined_sim

        # Limit cache size
        if len(similarity_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(similarity_cache.keys())[:200]
            for k in keys_to_remove:
                del similarity_cache[k]

        # Update similarity cache
        self.update_state("similarity_cache", similarity_cache)

        return combined_sim

    async def scale_token_limit_by_relevance(
        self,
        result: Dict,
        query_embedding: List[float],
        pdv: Optional[List[float]] = None,
    ) -> int:
        """Scale the token limit for a result based on its relevance to the query and PDV"""
        base_token_limit = self.valves.MAX_RESULT_TOKENS

        # Default to base if no similarity available
        if "similarity" not in result:
            return base_token_limit

        # Get the similarity score
        similarity = result.get("similarity", 0.5)

        # Calculate PDV alignment if available
        pdv_alignment = 0.5  # Neutral default
        if pdv is not None:
            try:
                # Get result content embedding
                content = result.get("content", "")
                content_embedding = await self.get_embedding(content[:2000])

                if content_embedding:
                    # Calculate alignment with PDV
                    alignment = np.dot(content_embedding, pdv)
                    pdv_alignment = (alignment + 1) / 2  # Normalize to 0-1
            except Exception as e:
                logger.error(f"Error calculating PDV alignment: {e}")

        # Combine similarity and PDV alignment
        combined_relevance = (similarity * 0.7) + (pdv_alignment * 0.3)

        # Scale between 0.5x and 1.5x of base limit
        scaling_factor = 0.5 + (combined_relevance * 1.0)  # Range: 0.5 to 1.5
        scaled_limit = int(base_token_limit * scaling_factor)

        # Cap at reasonable minimum and maximum
        min_limit = int(base_token_limit * 0.5)  # 50% of base
        max_limit = int(base_token_limit * 1.5)  # 150% of base

        scaled_limit = max(min_limit, min(max_limit, scaled_limit))

        logger.info(
            f"Scaled token limit for result: {scaled_limit} tokens "
            f"(similarity: {similarity:.2f}, scaling factor: {scaling_factor:.2f})"
        )

        return scaled_limit

    async def calculate_preference_impact(self, kept_items, removed_items, all_topics):
        """Calculate the impact of user preferences based on the proportion modified"""
        if not kept_items or not removed_items:
            return 0.0

        # Calculate impact based on proportion of items removed
        total_items = len(all_topics)
        if total_items == 0:
            return 0.0

        impact = len(removed_items) / total_items
        logger.info(
            f"User preference impact: {impact:.3f} ({len(removed_items)}/{total_items} items removed)"
        )
        return impact

    async def summarize_preferences_from_items(
        self, kept_items: List[str], removed_items: List[str]
    ) -> Optional[str]:
        """Summarize user preferences from kept and removed items using the LLM"""
        if not kept_items and not removed_items:
            return None

        try:
            max_examples = 10
            kept_sample = kept_items[:max_examples]
            removed_sample = removed_items[:max_examples]

            kept_block = "\n".join([f"- {item}" for item in kept_sample]) or "None"
            removed_block = (
                "\n".join([f"- {item}" for item in removed_sample]) or "None"
            )

            synthesis_model = self.get_synthesis_model()

            system_prompt = (
                "You analyze how a user selects or rejects research topics.\n"
                "From the kept vs removed examples, infer what kinds of topics the user prefers.\n"
                "Respond in the same language as most of the items (for example Polish or English).\n"
                "Return 1 to 3 short labels, 2–5 words each, separated by commas.\n"
                "Do not add explanations, numbering, or extra text."
            )

            user_prompt = (
                "Here are examples of topics the user decided to keep or remove.\n\n"
                "Kept topics:\n"
                f"{kept_block}\n\n"
                "Removed topics:\n"
                f"{removed_block}\n\n"
                "Based on these examples, describe the user's main preferences as short labels "
                "separated by commas."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.generate_completion(
                synthesis_model,
                messages,
                temperature=self.valves.SYNTHESIS_TEMPERATURE,
                user_facing=True,
            )

            if not response or "choices" not in response or not response["choices"]:
                return None

            raw_text = response["choices"][0]["message"]["content"]
            if not raw_text:
                return None

            first_line = raw_text.split("\n")[0].strip()
            first_line = first_line.lstrip("-•*").strip()
            if (
                len(first_line) >= 2
                and first_line[0] in ("'", '"')
                and first_line[-1] == first_line[0]
            ):
                first_line = first_line[1:-1].strip()

            return first_line or None
        except Exception as e:
            logger.error(f"Error summarizing user preferences from items: {e}")
            return None

    async def calculate_preference_direction_vector(
        self, kept_items: List[str], removed_items: List[str], all_topics: List[str]
    ) -> Dict:
        """Calculate the Preference Direction Vector based on kept and removed items"""
        if not kept_items or not removed_items:
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        # Get embeddings for kept and removed items in parallel
        kept_embeddings = []
        removed_embeddings = []

        # Get embeddings for kept items sequentially
        kept_embeddings = []
        for item in kept_items:
            embedding = await self.get_embedding(item)
            if embedding:
                kept_embeddings.append(embedding)

        removed_embeddings = []
        for item in removed_items:
            embedding = await self.get_embedding(item)
            if embedding:
                removed_embeddings.append(embedding)

        if not kept_embeddings or not removed_embeddings:
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        try:
            # Calculate mean vectors
            kept_mean = np.mean(kept_embeddings, axis=0)
            removed_mean = np.mean(removed_embeddings, axis=0)

            # Check for NaN or Inf values
            if (
                np.isnan(kept_mean).any()
                or np.isnan(removed_mean).any()
                or np.isinf(kept_mean).any()
                or np.isinf(removed_mean).any()
            ):
                logger.warning("Invalid values in kept or removed mean vectors")
                return {"pdv": None, "strength": 0.0, "impact": 0.0}

            # Calculate the preference direction vector
            pdv = kept_mean - removed_mean

            # Normalize the vector
            pdv_norm = np.linalg.norm(pdv)
            if pdv_norm < 1e-10:
                logger.warning("PDV has near-zero norm")
                return {"pdv": None, "strength": 0.0, "impact": 0.0}

            pdv = pdv / pdv_norm

            # Calculate preference strength (distance between centroids)
            strength = np.linalg.norm(kept_mean - removed_mean)

            # Calculate impact factor based on proportion of items removed
            impact = await self.calculate_preference_impact(
                kept_items, removed_items, all_topics
            )

            result = {
                "pdv": pdv.tolist(),
                "strength": float(strength),
                "impact": impact,
            }

            try:
                preference_labels = await self.summarize_preferences_from_items(
                    kept_items, removed_items
                )
            except Exception as e:
                logger.error(f"Error generating preference labels: {e}")
                preference_labels = None

            if preference_labels:
                result["labels"] = preference_labels

            return result
        except Exception as e:
            logger.error(f"Error calculating PDV: {e}")
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

    async def translate_pdv_to_words(self, pdv):
        """Translate a Preference Direction Vector (PDV) into human-readable concepts using stored preference labels"""
        if not pdv:
            return None

        try:
            state = self.get_state()
            user_preferences = state.get(
                "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
            )

            labels = user_preferences.get("labels")
            if labels:
                return labels

            feedback = state.get("last_preference_feedback")
            if feedback:
                labels = await self.summarize_preferences_from_items(
                    feedback.get("kept_items", []), feedback.get("removed_items", [])
                )
                if labels:
                    user_preferences["labels"] = labels
                    self.update_state("user_preferences", user_preferences)
                return labels

            return None
        except Exception as e:
            logger.error(f"Error translating PDV to words: {e}")
            return None

    async def translate_dimensions_to_words(self, dimensions, coverage):
        """Translate research dimensions to human-readable concepts using the LLM.

        For each dimension we:
        - find the topics with the strongest projection on that dimension,
        - pick one representative and a few nearest neighbours,
        - ask the LLM to name the common theme in 2–4 words.
        """
        if not dimensions or not coverage:
            return []

        # Get state for caching
        state = self.get_state()
        dimensions_cache = state.get("dimensions_translation_cache", {})

        # Create a unique cache key based on dimensions and coverage
        dim_hash = hash(str(dimensions.get("eigenvectors", [])[:3]))
        coverage_hash = hash(str(coverage))
        cache_key = f"dim_{dim_hash}_{coverage_hash}"

        # Check if we have a cached translation
        if cache_key in dimensions_cache:
            logger.info("Using cached dimension translation")
            return dimensions_cache[cache_key]

        dimension_labels = []

        eigenvectors = np.array(dimensions.get("eigenvectors", []))
        topic_texts = dimensions.get("topic_texts", [])
        topic_embeddings = dimensions.get("topic_embeddings", [])

        if (
            len(eigenvectors) == 0
            or len(eigenvectors) != len(coverage)
            or not topic_texts
            or not topic_embeddings
        ):
            default_labels = [
                {
                    "dimension": i + 1,
                    "words": f"Dimension {i+1}",
                    "coverage": coverage[i],
                }
                for i in range(len(coverage))
            ]
            dimensions_cache[cache_key] = default_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)
            return default_labels

        try:
            embeddings_array = np.array(topic_embeddings)

            # Ensure shape compatibility
            if embeddings_array.shape[1] != eigenvectors.shape[1]:
                logger.warning(
                    "Embedding dimension mismatch in translate_dimensions_to_words, "
                    "falling back to default labels"
                )
                default_labels = [
                    {
                        "dimension": i + 1,
                        "words": f"Dimension {i+1}",
                        "coverage": coverage[i],
                    }
                    for i in range(len(coverage))
                ]
                dimensions_cache[cache_key] = default_labels
                self.update_state("dimensions_translation_cache", dimensions_cache)
                return default_labels

            # Use synthesis model (the one used for summaries) for naming
            synthesis_model = self.get_synthesis_model()
            # Keep internal dimension naming prompts in English regardless of UI language
            label_lang = "en"

            for i, eigen_vector in enumerate(eigenvectors):
                # Project topics onto this dimension
                projection_scores = embeddings_array.dot(eigen_vector)

                # Sort topics by absolute projection strength (most representative first)
                indices = np.argsort(-np.abs(projection_scores))

                top_indices = indices[:5] if len(indices) >= 5 else indices
                if len(top_indices) == 0:
                    label_text = f"Dimension {i+1}"
                else:
                    representative = topic_texts[top_indices[0]]
                    neighbours = [
                        topic_texts[idx] for idx in top_indices[1:] if idx < len(topic_texts)
                    ]

                    topics_block_lines = [f"- {representative}"]
                    for n in neighbours:
                        topics_block_lines.append(f"- {n}")
                    topics_block = "\n".join(topics_block_lines)

                    system_prompt = (
                        "You are helping to name semantic clusters of research topics.\n"
                        "Given a small list of closely related topics, identify the single common theme "
                        "and express it as a short label of 2–4 words in English.\n"
                        "Return only the label text, without quotation marks, numbering, or extra commentary."
                    )
                    user_prompt = (
                        "Here is a cluster of related research topics:\n"
                        f"{topics_block}\n\n"
                        "Name the shared theme in 2–4 words in English."
                    )

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    try:
                        response = await self.generate_completion(
                            synthesis_model,
                            messages,
                            temperature=self.valves.SYNTHESIS_TEMPERATURE,
                            user_facing=True,
                        )
                        raw_label = (
                            response["choices"][0]["message"]["content"]
                            if response
                            else ""
                        )
                        # Take first line, strip bullets/quotes
                        label_line = raw_label.split("\n")[0].strip()
                        label_line = label_line.lstrip("-•*").strip()
                        if (
                            len(label_line) >= 2
                            and label_line[0] in ("'", '"')
                            and label_line[-1] == label_line[0]
                        ):
                            label_line = label_line[1:-1].strip()

                        label_text = label_line or f"Dimension {i+1}"
                    except Exception as e:
                        logger.error(f"Error generating LLM label for dimension {i+1}: {e}")
                        label_text = f"Dimension {i+1}"

                cov_percentage = coverage[i]
                dimension_labels.append(
                    {
                        "dimension": i + 1,
                        "words": label_text,
                        "coverage": cov_percentage,
                    }
                )

            dimensions_cache[cache_key] = dimension_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)

            return dimension_labels
        except Exception as e:
            logger.error(f"Error translating dimensions to words with LLM: {e}")
            default_labels = [
                {
                    "dimension": i + 1,
                    "words": f"Dimension {i+1}",
                    "coverage": coverage[i],
                }
                for i in range(len(coverage))
            ]
            dimensions_cache[cache_key] = default_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)
            return default_labels

    async def calculate_preference_alignment(self, content_embedding, pdv):
        """Calculate alignment between content and preference vector"""
        if not pdv or not content_embedding:
            return 0.5  # Neutral value if we can't calculate

        try:
            # Calculate dot product between vectors
            alignment = np.dot(content_embedding, pdv)

            # Normalize to 0-1 scale (dot product is between -1 and 1)
            normalized = (alignment + 1) / 2

            return normalized
        except Exception as e:
            logger.error(f"Error calculating preference alignment: {e}")
            return 0.5  # Neutral value on error

    async def update_research_dimensions_display(self):
        """Ensure research dimensions are properly updated for display"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")

        if research_dimensions:
            # Don't make a separate copy - just point to the actual dimension coverage
            coverage = research_dimensions.get("coverage", [])
            if coverage:
                self.update_state("latest_dimension_coverage", coverage)
                logger.info(
                    f"Updated latest dimension coverage with {len(coverage)} values"
                )
            else:
                logger.warning("Research dimensions exist but coverage is empty")
        else:
            logger.warning("No research dimensions available for display")

    async def initialize_research_dimensions(
        self, outline_items: List[str], user_query: str
    ):
        """Initialize the semantic dimensions for tracking research progress.

        In addition to the PCA decomposition itself, this stores the
        outline texts and their embeddings so we can later derive
        human-readable names for each dimension using the LLM.
        """
        try:
            topic_texts: List[str] = []
            topic_embeddings: List[List[float]] = []

            # Get embeddings for each outline item sequentially
            for item in outline_items:
                embedding = await self.get_embedding(item[:2000])
                if embedding:
                    topic_texts.append(item)
                    topic_embeddings.append(embedding)

            # Ensure we have enough embeddings for PCA
            if len(topic_embeddings) < 3:
                logger.warning(
                    f"Not enough valid embeddings for research dimensions: {len(topic_embeddings)}/3 required"
                )
                self.update_state("research_dimensions", None)
                return

            # Apply PCA to reduce to key dimensions
            pca = PCA(n_components=min(10, len(topic_embeddings)))
            embedding_array = np.array(topic_embeddings)
            pca.fit(embedding_array)

            # Store the PCA model, topic metadata, and progress trackers
            research_dimensions = {
                "eigenvectors": pca.components_.tolist(),
                "eigenvalues": pca.explained_variance_.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "total_variance": pca.explained_variance_ratio_.sum(),
                "dimensions": pca.n_components_,
                "coverage": np.zeros(
                    pca.n_components_
                ).tolist(),  # Initialize empty coverage
                # For LLM-based naming of dimensions
                "topic_texts": topic_texts,
                "topic_embeddings": topic_embeddings,
            }

            self.update_state("research_dimensions", research_dimensions)

            # Immediately store a copy of coverage for display
            self.update_state(
                "latest_dimension_coverage", research_dimensions["coverage"]
            )

            logger.info(
                f"Initialized research dimensions with {pca.n_components_} dimensions"
            )
        except Exception as e:
            logger.error(f"Error initializing research dimensions: {e}")
            self.update_state("research_dimensions", None)

    async def update_dimension_coverage(
        self, content: str, quality_factor: float = 1.0
    ):
        """Update the coverage of research dimensions based on new content"""
        # Get current state
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return

        try:
            # Get embedding for the content
            content_embedding = await self.get_embedding(content[:2000])
            if not content_embedding:
                return

            # Get current coverage
            current_coverage = research_dimensions.get("coverage", [])
            eigenvectors = research_dimensions.get("eigenvectors", [])

            if not current_coverage or not eigenvectors:
                return

            # Convert to numpy for calculations
            coverage_array = np.array(current_coverage)
            eigenvectors_array = np.array(eigenvectors)

            # Calculate projection and contribution
            projection = np.dot(np.array(content_embedding), eigenvectors_array.T)
            contribution = np.abs(projection) * quality_factor

            # Update coverage directly
            for i in range(min(len(contribution), len(coverage_array))):
                current_value = coverage_array[i]
                new_contribution = contribution[i] * (1 - current_value / 2)
                coverage_array[i] += new_contribution

            # Update the coverage in research_dimensions
            research_dimensions["coverage"] = coverage_array.tolist()

            # Update both state keys
            self.update_state("research_dimensions", research_dimensions)
            self.update_state("latest_dimension_coverage", coverage_array.tolist())

            logger.debug(
                f"Updated dimension coverage: {[round(c * 100) for c in coverage_array.tolist()]}%"
            )

        except Exception as e:
            logger.error(f"Error updating dimension coverage: {e}")

    async def identify_research_gaps(self) -> List[str]:
        """Identify semantic dimensions that need more research"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return []

        try:
            # Find dimensions with low coverage
            coverage = np.array(research_dimensions["coverage"])

            # Sort dimensions by coverage (ascending)
            sorted_dims = np.argsort(coverage)

            # Return indices of the least covered dimensions (lowest 3 that are below 50% coverage)
            gaps = [i for i in sorted_dims[:3] if coverage[i] < 0.5]

            return gaps
        except Exception as e:
            logger.error(f"Error identifying research gaps: {e}")
            return []

    def _looks_like_metadata_query(self, text: str) -> bool:
        """Heuristic filter for strings that are likely API metadata, not real search queries."""
        if not text:
            return True

        value = text.strip()
        lower = value.lower()

        # Obvious metadata keys or boilerplate
        meta_tokens = {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "role",
            "content",
            "system_fingerprint",
            "index",
        }
        if lower in meta_tokens:
            return True

        # Very short single words are unlikely to be good queries
        if " " not in lower and len(lower) <= 3:
            return True

        # Model / engine names and similar identifiers
        engine_markers = ["gpt-oss", "gpt-", "gpt4", "gpt-4", "ollama", "text-davinci"]
        if any(marker in lower for marker in engine_markers):
            return True

        # Likely UUIDs or hex IDs
        import re

        if re.fullmatch(r"[0-9a-f\-]{16,}", lower):
            return True

        return False

    def _extract_queries_from_llm_response(
        self, query_content: str, user_message: str, max_queries: int = 3
    ) -> List[str]:
        """Robust fallback to extract reasonable query-like strings from an LLM response."""
        import re

        candidates: List[str] = []

        # 1) Prefer bullet/numbered list lines – typical pattern for queries
        for line in query_content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
                candidate = re.sub(r"^([-*]|\d+\.)\s*", "", stripped)
                if candidate and not self._looks_like_metadata_query(candidate):
                    candidates.append(candidate)

        # 2) If that fails, fall back to quoted strings but filter aggressively
        if not candidates:
            all_strings = re.findall(r'"([^"]+)"', query_content)
            for s in all_strings:
                if self._looks_like_metadata_query(s):
                    continue
                candidates.append(s)

        # 3) Last-resort fallback: generic query from the original user message
        if not candidates:
            return [f"Information about {user_message}"]

        # De-duplicate while preserving order
        seen = set()
        cleaned: List[str] = []
        for c in candidates:
            normalized = c.strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            cleaned.append(normalized)

        return cleaned[:max_queries]
