from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeCompressionMixin:
    async def compress_content_with_local_similarity(
        self,
        content: str,
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
        ratio: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Apply semantic compression with local similarity influence and token limiting"""
        start_time = time.perf_counter() if getattr(self.valves, "DEBUG_TIMING", False) else None
        def _finish(val: str) -> str:
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info("TIMING compress_local %.3fs", elapsed)
            return val
        # Skip compression for very short content
        if len(content) < 100:
            return _finish(content)

        # Apply token limit if specified
        if max_tokens:
            content_tokens = await self.count_tokens(content)
            if content_tokens <= max_tokens:
                return _finish(content)

            # If over limit, use token-based compression ratio
            if not ratio:
                ratio = max_tokens / content_tokens

        # Split content into chunks based on chunk_level
        chunks = self.chunk_text(content)

        # Skip compression if only one chunk
        if len(chunks) <= 1:
            return _finish(content)

        # Get embeddings for chunks sequentially
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            if embedding:
                chunk_embeddings.append(embedding)

        # Skip compression if not enough embeddings
        if len(chunk_embeddings) <= 1:
            return _finish(content)

        # Define compression ratio if not provided
        if ratio is None:
            level = self.valves.COMPRESSION_LEVEL
            ratio = COMPRESSION_RATIO_MAP.get(level, 0.5)

        # Calculate how many chunks to keep
        n_chunks = len(chunk_embeddings)
        n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

        # Ensure we're compressing at least a little
        if n_keep >= n_chunks:
            n_keep = max(1, n_chunks - 1)

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(chunk_embeddings)

            # Calculate document centroid
            document_centroid = np.mean(embeddings_array, axis=0)

            # Calculate local similarity for each chunk
            local_similarities = []
            local_radius = self.valves.LOCAL_INFLUENCE_RADIUS  # Get from valve

            for i in range(len(embeddings_array)):
                # Calculate similarity to adjacent chunks (local influence)
                local_sim = 0.0
                count = 0

                # Check previous chunks within radius
                for j in range(max(0, i - local_radius), i):
                    local_sim += cosine_similarity(
                        [embeddings_array[i]], [embeddings_array[j]]
                    )[0][0]
                    count += 1

                # Check next chunks within radius
                for j in range(i + 1, min(len(embeddings_array), i + local_radius + 1)):
                    local_sim += cosine_similarity(
                        [embeddings_array[i]], [embeddings_array[j]]
                    )[0][0]
                    count += 1

                if count > 0:
                    local_sim /= count

                local_similarities.append(local_sim)

            # Calculate importance scores with all factors
            importance_scores = []
            state = self.get_state()
            user_preferences = state.get(
                "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
            )

            for i, embedding in enumerate(embeddings_array):
                # Fix any NaN or Inf values
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    embedding = np.nan_to_num(
                        embedding, nan=0.0, posinf=1.0, neginf=-1.0
                    )

                # Calculate similarity to document centroid
                doc_similarity = cosine_similarity([embedding], [document_centroid])[0][
                    0
                ]

                # Calculate similarity to query
                query_similarity = cosine_similarity([embedding], [query_embedding])[0][
                    0
                ]

                # Calculate similarity to previous summary if provided
                summary_similarity = 0.0
                if summary_embedding is not None:
                    summary_similarity = cosine_similarity(
                        [embedding], [summary_embedding]
                    )[0][0]
                    # Blend query and summary similarity
                    query_similarity = (
                        query_similarity * self.valves.FOLLOWUP_WEIGHT
                    ) + (summary_similarity * (1.0 - self.valves.FOLLOWUP_WEIGHT))

                # Include local similarity influence
                local_influence = local_similarities[i]

                # Include preference direction vector if available
                pdv_alignment = 0.5  # Neutral default
                if (
                    self.valves.USER_PREFERENCE_THROUGHOUT
                    and user_preferences["pdv"] is not None
                ):
                    chunk_embedding_np = np.array(embedding)
                    pdv_np = np.array(user_preferences["pdv"])
                    alignment = np.dot(chunk_embedding_np, pdv_np)
                    pdv_alignment = (alignment + 1) / 2  # Normalize to 0-1

                    # Weight by preference strength
                    pdv_influence = min(0.3, user_preferences["strength"] / 10)
                else:
                    pdv_influence = 0.0

                # Weight the factors
                doc_weight = (
                    1.0 - self.valves.QUERY_WEIGHT
                ) * 0.4  # Some preference towards relevance towards query
                local_weight = (
                    1.0 - self.valves.QUERY_WEIGHT
                ) * 0.8  # More preference towards standout local chunks
                query_weight = self.valves.QUERY_WEIGHT * (1.0 - pdv_influence)

                final_score = (
                    (doc_similarity * doc_weight)
                    + (query_similarity * query_weight)
                    + (local_influence * local_weight)
                    + (pdv_alignment * pdv_influence)
                )

                importance_scores.append((i, final_score))

            # Sort chunks by importance (most important first)
            importance_scores.sort(key=lambda x: x[1], reverse=True)

            # Select the top n_keep most important chunks
            selected_indices = [x[0] for x in importance_scores[:n_keep]]

            # Sort indices to maintain original document order
            selected_indices.sort()

            # Get the selected chunks
            selected_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]

            # Join compressed chunks back into text with proper formatting
            chunk_level = self.valves.CHUNK_LEVEL
            if chunk_level == 1:  # Phrase level
                compressed_content = " ".join(selected_chunks)
            elif chunk_level == 2:  # Sentence level
                processed_sentences = []
                for sentence in selected_chunks:
                    if not sentence.endswith((".", "!", "?", ":", ";")):
                        sentence += "."
                    processed_sentences.append(sentence)
                compressed_content = " ".join(processed_sentences)
            else:  # Paragraph levels
                compressed_content = "\n".join(selected_chunks)

            # Verify token count if max_tokens specified
            if max_tokens:
                final_tokens = await self.count_tokens(compressed_content)

                # If still over limit, apply additional compression
                if final_tokens > max_tokens:
                    # Calculate new ratio based on tokens
                    new_ratio = max_tokens / final_tokens
                    # Recursively compress with more aggressive ratio
                    compressed_content = (
                        await self.compress_content_with_local_similarity(
                            compressed_content,
                            query_embedding,
                            summary_embedding,
                            ratio=new_ratio,
                        )
                    )

            return _finish(compressed_content)

        except Exception as e:
            logger.error(f"Error during compression with local similarity: {e}")

            # If max_tokens specified and error occurred, do basic truncation
            if max_tokens and content:
                # Estimate character position based on token limit
                content_tokens = await self.count_tokens(content)
                if content_tokens > max_tokens:
                    char_ratio = max_tokens / content_tokens
                    char_limit = int(len(content) * char_ratio)
                    return _finish(content[:char_limit])

            return _finish(content)

    async def compress_content_with_eigendecomposition(
        self,
        content: str,
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
        ratio: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Apply semantic compression using eigendecomposition with token limiting"""
        start_time = time.perf_counter() if getattr(self.valves, "DEBUG_TIMING", False) else None
        def _finish(val: str) -> str:
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info("TIMING compress_eigen %.3fs", elapsed)
            return val
        # Skip compression for very short content
        if len(content) < 200:
            return _finish(content)

        # Apply token limit if specified
        if max_tokens:
            content_tokens = await self.count_tokens(content)
            if content_tokens <= max_tokens:
                return _finish(content)

            # If over limit, use token-based compression ratio
            if not ratio:
                ratio = max_tokens / content_tokens

        # Split content into chunks based on chunk_level
        chunks = self.chunk_text(content)

        # Skip compression if only one chunk
        if len(chunks) <= 2:
            return _finish(content)

        # Get embeddings for chunks sequentially
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            if embedding:
                chunk_embeddings.append(embedding)

        # Skip compression if not enough embeddings
        if len(chunk_embeddings) <= 2:
            return _finish(content)

        # Define compression ratio if not provided
        if ratio is None:
            level = self.valves.COMPRESSION_LEVEL
            ratio = COMPRESSION_RATIO_MAP.get(level, 0.5)

        # Calculate how many chunks to keep
        n_chunks = len(chunks)
        n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

        # Ensure we're compressing at least a little
        if n_keep >= n_chunks:
            n_keep = max(1, n_chunks - 1)

        # Debug instrumentation to help trace eigendecomposition issues
        if logger.isEnabledFor(logging.DEBUG):
            try:
                nan_inf = any(
                    np.isnan(np.array(emb)).any() or np.isinf(np.array(emb)).any()
                    for emb in chunk_embeddings
                )
            except Exception:
                nan_inf = True
            logger.debug(
                "Eigencompression debug: chunks=%d chunk_embeddings=%d n_keep=%d ratio=%.3f nan_or_inf=%s",
                len(chunks),
                len(chunk_embeddings),
                n_keep,
                ratio,
                nan_inf,
            )

        try:
            # Pull cached semantic transformations once (used for query transforms)
            semantic_transformations = self.get_state().get("semantic_transformations")

            # Perform semantic eigendecomposition
            eigendecomposition = await self.compute_semantic_eigendecomposition(
                chunks, chunk_embeddings
            )

            # Validate eigendecomposition structure before proceeding
            required_keys = {
                "projected_embeddings",
                "eigenvectors",
                "explained_variance",
                "n_components",
            }
            if not isinstance(eigendecomposition, dict) or not required_keys.issubset(
                eigendecomposition.keys()
            ):
                logger.debug(
                    "Eigendecomposition missing keys: present=%s required=%s raw=%s",
                    list(eigendecomposition.keys()) if isinstance(eigendecomposition, dict) else type(eigendecomposition),
                    list(required_keys),
                    eigendecomposition,
                )
                logger.warning(
                    "Eigendecomposition invalid or incomplete; using local similarity compression"
                )
                return _finish(
                    await self.compress_content_with_local_similarity(
                        content, query_embedding, summary_embedding, ratio, max_tokens
                    )
                )

            if eigendecomposition:
                # Calculate importance scores based on the eigendecomposition
                embeddings_array = np.array(chunk_embeddings)
                importance_scores = []

                # Create basic directions
                directions = {}
                if query_embedding:
                    directions["query"] = query_embedding
                if summary_embedding:
                    directions["summary"] = summary_embedding

                state = self.get_state()
                user_preferences = state.get(
                    "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
                )
                if user_preferences["pdv"] is not None:
                    directions["pdv"] = user_preferences["pdv"]

                # Create transformation
                transformation = await self.create_semantic_transformation(
                    eigendecomposition,
                    pdv=(
                        user_preferences["pdv"]
                        if user_preferences["impact"] > 0.1
                        else None
                    ),
                )

                # Project chunks into the principal component space for better analysis
                raw_projected_chunks = eigendecomposition["projected_embeddings"]

                if not isinstance(raw_projected_chunks, list) or not raw_projected_chunks:
                    logger.debug(
                        "Projected embeddings malformed: type=%s sample=%s",
                        type(raw_projected_chunks),
                        raw_projected_chunks[:3] if isinstance(raw_projected_chunks, list) else raw_projected_chunks,
                    )
                    logger.warning(
                        "Projected embeddings malformed; using local similarity compression"
                    )
                    return _finish(
                        await self.compress_content_with_local_similarity(
                            content,
                            query_embedding,
                            summary_embedding,
                            ratio,
                            max_tokens,
                        )
                    )

                # Validate column count; if mismatched, fall back rather than padding
                first_pc = raw_projected_chunks[0]
                if isinstance(first_pc, (int, float)):
                    col_count = 1
                elif isinstance(first_pc, (list, tuple, np.ndarray)):
                    col_count = len(first_pc)
                else:
                    col_count = 0

                expected_components = eigendecomposition["n_components"]
                if col_count < expected_components:
                    logger.warning(
                        "Projected embeddings have fewer columns (%s) than expected components (%s); using local similarity compression",
                        col_count,
                        expected_components,
                    )
                    return _finish(
                        await self.compress_content_with_local_similarity(
                            content,
                            query_embedding,
                            summary_embedding,
                            ratio,
                            max_tokens,
                        )
                    )

                eigenvectors = np.array(eigendecomposition["eigenvectors"])
                projected_chunks = self._normalize_projected_chunks(
                    raw_projected_chunks, expected_components
                )

                # Calculate local coherence using the eigenspace
                local_coherence = []
                local_radius = self.valves.LOCAL_INFLUENCE_RADIUS

                for i in range(len(projected_chunks)):
                    # Calculate similarity to adjacent chunks
                    local_sim = 0.0
                    count = 0

                    # Look at previous and next chunks within radius
                    for j in range(
                        max(0, i - local_radius),
                        min(len(projected_chunks), i + local_radius + 1),
                    ):
                        if i == j:
                            continue

                        # Use weighted similarity in eigenspace
                        sim = 0.0
                        for k in range(eigendecomposition["n_components"]):
                            # Weight by eigenvalue importance
                            weight = eigendecomposition["explained_variance"][k]
                            dim_sim = 1.0 - abs(
                                projected_chunks[i][k] - projected_chunks[j][k]
                            )
                            sim += weight * dim_sim

                        local_sim += sim
                        count += 1

                    if count > 0:
                        local_sim /= count
                    local_coherence.append(local_sim)

                # Calculate relevance to query using transformed embeddings
                if query_embedding:
                    try:
                        transformed_query = query_embedding

                        # Ensure we're getting transformed embeddings if a transformation is available
                        if semantic_transformations:
                            transformed_candidate = (
                                await self.apply_semantic_transformation(
                                    query_embedding, semantic_transformations
                                )
                            )
                            if transformed_candidate:
                                transformed_query = transformed_candidate

                        # Calculate similarities with transformed query in one operation
                        query_relevance = []
                        for chunk_embedding in chunk_embeddings:
                            if chunk_embedding:
                                # Get similarity to transformed query
                                similarity = cosine_similarity(
                                    [chunk_embedding], [transformed_query]
                                )[0][0]
                                query_relevance.append(similarity)
                            else:
                                query_relevance.append(
                                    0.5
                                )  # Default for missing embeddings
                    except Exception as e:
                        logger.warning(f"Error calculating query relevance: {e}")
                        query_relevance = [0.5] * len(projected_chunks)
                else:
                    # Default relevance if no query
                    query_relevance = [0.5] * len(projected_chunks)

                # Combine scores
                for i in range(len(chunks)):
                    if i >= len(local_coherence) or i >= len(query_relevance):
                        continue

                    # Weights for different factors
                    coherence_weight = 0.4
                    relevance_weight = 0.6

                    # Adjust based on user preferences
                    if (
                        user_preferences["pdv"] is not None
                        and user_preferences["impact"] > 0.1
                    ):
                        # Reduce other weights to make room for preference weight
                        pdv_weight = min(0.3, user_preferences["impact"])
                        coherence_weight *= 1.0 - pdv_weight
                        relevance_weight *= 1.0 - pdv_weight

                        # Calculate PDV alignment if available
                        if i < len(chunk_embeddings):
                            try:
                                chunk_embed = chunk_embeddings[i]
                                pdv_alignment = np.dot(
                                    chunk_embed, user_preferences["pdv"]
                                )
                                # Normalize to 0-1 range
                                pdv_alignment = (pdv_alignment + 1) / 2
                            except Exception as e:
                                logger.warning(f"Error calculating PDV alignment: {e}")
                                pdv_alignment = 0.5
                        else:
                            pdv_alignment = 0.5

                        final_score = (
                            (local_coherence[i] * coherence_weight)
                            + (query_relevance[i] * relevance_weight)
                            + (pdv_alignment * pdv_weight)
                        )
                    else:
                        final_score = (local_coherence[i] * coherence_weight) + (
                            query_relevance[i] * relevance_weight
                        )

                    importance_scores.append((i, final_score))

                # Sort chunks by importance
                importance_scores.sort(key=lambda x: x[1], reverse=True)

                # Select the top n_keep chunks
                selected_indices = [x[0] for x in importance_scores[:n_keep]]

                # Sort to maintain document order
                selected_indices.sort()

                # Get selected chunks
                selected_chunks = [
                    chunks[i] for i in selected_indices if i < len(chunks)
                ]

                # Join compressed chunks with proper formatting
                chunk_level = self.valves.CHUNK_LEVEL
                if chunk_level == 1:  # Phrase level
                    compressed_content = " ".join(selected_chunks)
                elif chunk_level == 2:  # Sentence level
                    processed_sentences = []
                    for sentence in selected_chunks:
                        if not sentence.endswith((".", "!", "?", ":", ";")):
                            sentence += "."
                        processed_sentences.append(sentence)
                    compressed_content = " ".join(processed_sentences)
                else:  # Paragraph levels
                    compressed_content = "\n".join(selected_chunks)

                # Verify token count if max_tokens specified
                if max_tokens:
                    final_tokens = await self.count_tokens(compressed_content)

                    # If still over limit, apply additional compression
                    if final_tokens > max_tokens:
                        # Calculate new ratio based on tokens
                        new_ratio = max_tokens / final_tokens
                        # Recursively compress with more aggressive ratio
                        compressed_content = (
                            await self.compress_content_with_eigendecomposition(
                                compressed_content,
                                query_embedding,
                                summary_embedding,
                                ratio=new_ratio,
                            )
                        )

                return _finish(compressed_content)

            # Fallback if eigendecomposition fails
            logger.warning(
                "Eigendecomposition compression failed, using original method"
            )
            return _finish(
                await self.compress_content_with_local_similarity(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )
            )

        except Exception as e:
            # Add contextual debug info to help pinpoint the root cause
            logger.error(
                "Error during compression with eigendecomposition: %s | chunks=%d embeddings=%d n_keep=%d ratio=%s",
                e,
                len(chunks),
                len(chunk_embeddings),
                n_keep,
                ratio,
            )
            # Fall back to original compression method
            try:
                return _finish(
                    await self.compress_content_with_local_similarity(
                        content, query_embedding, summary_embedding, ratio, max_tokens
                    )
                )
            except Exception as fallback_error:
                logger.error(f"Fallback compression also failed: {fallback_error}")

                # If max_tokens specified and all compression failed, do basic truncation
                if max_tokens and content:
                    # Estimate character position based on token limit
                    content_tokens = await self.count_tokens(content)
                    if content_tokens > max_tokens:
                        char_ratio = max_tokens / content_tokens
                        char_limit = int(len(content) * char_ratio)
                        return _finish(content[:char_limit])

                return _finish(content)  # Return original content if both methods fail

    async def handle_repeated_content(
        self, content: str, url: str, query_embedding: List[float], repeat_count: int
    ) -> str:
        """Process repeated content with improved sliding window and adaptive shrinkage"""
        state = self.get_state()
        url_selected_count = state.get("url_selected_count", {})
        url_token_counts = state.get("url_token_counts", {})

        # Only consider URLs that were actually shown to the user
        selected_count = url_selected_count.get(url, 0)

        # If first occurrence, return unchanged
        if selected_count < 1:
            total_tokens = await self.count_tokens(content)
            url_token_counts[url] = total_tokens
            self.update_state("url_token_counts", url_token_counts)
            return content

        # Get total tokens for this URL
        total_tokens = url_token_counts.get(url, 0)
        if total_tokens == 0:
            # Count if not already done
            total_tokens = await self.count_tokens(content)
            url_token_counts[url] = total_tokens
            self.update_state("url_token_counts", url_token_counts)

        # Calculate max window size
        max_tokens = self.valves.MAX_RESULT_TOKENS
        window_factor = self.valves.REPEAT_WINDOW_FACTOR

        # For any repeated result, decide whether to apply sliding window or compression/centering
        if total_tokens > max_tokens:
            # Large content - apply sliding window logic
            # Calculate window position based on repeat count and content size
            window_start = int((repeat_count - 1) * window_factor * max_tokens)

            # Check if we've reached the end of the content
            if window_start >= total_tokens:
                # We've cycled through once, now start shrinking
                cycles_completed = window_start // total_tokens

                # Calculate shrinkage: keep 70% for each full cycle completed
                shrink_factor = 0.7**cycles_completed

                # Calculate new window size with shrinkage
                window_size = int(max_tokens * shrink_factor)
                window_size = max(200, window_size)  # Set minimum window size

                # Recalculate start position for new cycle with smaller window
                window_start = window_start % total_tokens

                logger.info(
                    f"Repeat URL {url} (count: {selected_count}): applying shrinkage after full cycle. "
                    f"Factor: {shrink_factor:.2f}, window size: {window_size} tokens"
                )
            else:
                # Still sliding through content, use full window size
                window_size = max_tokens
                logger.info(
                    f"Repeat URL {url} (count: {selected_count}): sliding window, "
                    f"starting at token {window_start}, window size {window_size}"
                )

            # Extract window of tokens from content
            window_content = await self.extract_token_window(
                content, window_start, window_size
            )

            return window_content
        else:
            # Content already fits within max tokens - apply compression/centering
            logger.info(
                f"Repeat URL {url} (count: {selected_count}): applying compression/centering for content already within token limit"
            )

            # Get content embedding to find most relevant section
            content_embedding = await self.get_embedding(content[:2000])
            if not content_embedding:
                return content

            # Calculate relevance to query to identify most relevant portion
            try:
                # Get text chunks and their embeddings
                chunks = self.chunk_text(content)
                if len(chunks) <= 3:  # Not enough chunks to do meaningful re-centering
                    return content

                # Get chunk embeddings sequentially
                chunk_embeddings = []
                relevance_scores = []
                for i, chunk in enumerate(chunks):
                    chunk_embedding = await self.get_embedding(chunk[:2000])
                    if chunk_embedding:
                        chunk_embeddings.append(chunk_embedding)
                        relevance = cosine_similarity(
                            [chunk_embedding], [query_embedding]
                        )[0][0]
                        relevance_scores.append((i, relevance))

                # Sort by relevance
                relevance_scores.sort(key=lambda x: x[1], reverse=True)

                # Get most relevant chunk index
                if relevance_scores:
                    most_relevant_idx = relevance_scores[0][0]

                    # Re-center the window around the most relevant chunk
                    start_idx = max(0, most_relevant_idx - len(chunks) // 4)
                    end_idx = min(len(chunks), most_relevant_idx + len(chunks) // 4 + 1)

                    # Combine chunks to form re-centered content
                    recentered_content = "\n".join(chunks[start_idx:end_idx])
                    return recentered_content

            except Exception as e:
                logger.error(f"Error re-centering window: {e}")

            # Fallback to original content if re-centering fails
            return content

    async def apply_stepped_compression(
        self,
        results_history: List[Dict],
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Apply tiered compression to all research results based on age"""
        if not self.valves.STEPPED_SYNTHESIS_COMPRESSION or len(results_history) <= 2:
            return results_history

        # Make a copy to avoid modifying the original
        results = results_history.copy()

        # Divide results into first 50% (older) and second 50% (newer)
        mid_point = len(results) // 2
        older_results = results[:mid_point]
        newer_results = results[mid_point:]

        # Track token counts before and after compression
        total_tokens_before = 0
        total_tokens_after = 0

        # Define token limit for results
        max_tokens = self.valves.COMPRESSION_SETPOINT

        # Process older results with standard compression
        processed_older = []
        for result in older_results:
            content = result.get("content", "")
            url = result.get("url", "")

            # Count tokens in original content
            original_tokens = await self.count_tokens(content)
            total_tokens_before += original_tokens

            # Skip very short content
            if len(content) < 300:
                result["tokens"] = original_tokens
                processed_older.append(result)
                total_tokens_after += original_tokens
                continue

            # Apply standard compression
            compression_level = self.valves.COMPRESSION_LEVEL

            # Map compression level to ratio
            ratio = COMPRESSION_RATIO_MAP.get(compression_level, 0.5)

            try:
                # Compress using eigendecomposition with token limit
                compressed = await self.compress_content_with_eigendecomposition(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )

                # Count tokens in compressed content
                compressed_tokens = await self.count_tokens(compressed)
                total_tokens_after += compressed_tokens

                # Create new result with compressed content
                new_result = result.copy()
                new_result["content"] = compressed
                new_result["tokens"] = compressed_tokens

                # Log the token reduction
                logger.info(
                    f"Standard compression (older result): {original_tokens} → {compressed_tokens} tokens "
                    f"({compressed_tokens/original_tokens:.1%} of original)"
                )

                processed_older.append(new_result)

            except Exception as e:
                logger.error(f"Error during standard compression: {e}")
                # Keep original if compression fails
                result["tokens"] = original_tokens
                processed_older.append(result)
                total_tokens_after += original_tokens

        # Process newer results with more compression
        processed_newer = []
        for result in newer_results:
            content = result.get("content", "")
            url = result.get("url", "")

            # Count tokens in original content
            original_tokens = await self.count_tokens(content)
            total_tokens_before += original_tokens

            # Skip very short content
            if len(content) < 300:
                result["tokens"] = original_tokens
                processed_newer.append(result)
                total_tokens_after += original_tokens
                continue

            # Apply one level higher compression for newer results
            compression_level = min(10, self.valves.COMPRESSION_LEVEL + 1)

            # Map compression level to ratio
            ratio = COMPRESSION_RATIO_MAP.get(compression_level, 0.5)

            try:
                # Compress using eigendecomposition with token limit
                compressed = await self.compress_content_with_eigendecomposition(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )

                # Count tokens in compressed content
                compressed_tokens = await self.count_tokens(compressed)
                total_tokens_after += compressed_tokens

                # Create new result with compressed content
                new_result = result.copy()
                new_result["content"] = compressed
                new_result["tokens"] = compressed_tokens

                # Log the token reduction
                logger.info(
                    f"Higher compression (newer result): {original_tokens} → {compressed_tokens} tokens "
                    f"({compressed_tokens/original_tokens:.1%} of original)"
                )

                processed_newer.append(new_result)

            except Exception as e:
                logger.error(f"Error during higher compression: {e}")
                # Keep original if compression fails
                result["tokens"] = original_tokens
                processed_newer.append(result)
                total_tokens_after += original_tokens

        # Log the overall token reduction
        token_reduction = total_tokens_before - total_tokens_after
        if total_tokens_before > 0:
            percent_reduction = (token_reduction / total_tokens_before) * 100
            logger.info(
                f"Stepped compression total results: {total_tokens_before} → {total_tokens_after} tokens "
                f"(saved {token_reduction} tokens, {percent_reduction:.1f}% reduction)"
            )

        # Update memory statistics consistently
        await self.update_token_counts()

        # Combine and return all results in original order
        return processed_older + processed_newer

    def _normalize_projected_chunks(
        self, projected_chunks: Any, n_components: int
    ) -> List[List[float]]:
        """Ensure projected chunks are 2D and have at least n_components columns."""
        if not isinstance(projected_chunks, list):
            return []

        normalized: List[List[float]] = []
        for pc in projected_chunks:
            if isinstance(pc, (int, float)):
                row = [float(pc)]
            elif isinstance(pc, (list, tuple, np.ndarray)):
                row = [float(x) for x in pc]
            else:
                # Unknown shape; skip
                continue

            if len(row) < n_components:
                row = row + [0.0] * (n_components - len(row))
            normalized.append(row)
        return normalized
