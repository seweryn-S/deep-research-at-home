from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeSearchingMixin:
    async def process_search_result(
        self,
        result: Dict,
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> Dict:
        """Process a search result to extract and compress content with token limiting"""
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", "")

        # Require a URL for all results
        if not url:
            if self.valves.DEBUG_SEARCH:
                logger.info(
                    "SEARCH DEBUG: result_missing_url query=%r title=%r",
                    query,
                    title,
                )
            return {
                "title": title or f"Result for '{query}'",
                "url": "",
                "content": "This result has no associated URL and cannot be processed.",
                "query": query,
                "valid": False,
            }

        await self.emit_status("info", f"Processing result: {title[:50]}...", False)

        try:
            # Get state
            state = self.get_state()
            url_selected_count = state.get("url_selected_count", {})
            url_token_counts = state.get("url_token_counts", {})
            master_source_table = state.get("master_source_table", {})

            # Check if this is a repeated URL
            repeat_count = 0
            repeat_count = url_selected_count.get(url, 0)

            # If the snippet is empty or short but we have a URL, try to fetch content
            if (not snippet or len(snippet) < 200) and url:
                await self.emit_status(
                    "info", f"Fetching content from URL: {url}...", False
                )
                content = await self.content_processor.fetch_content(url)

                if content and len(content) > 200:
                    snippet = content
                    logger.debug(
                        f"Successfully fetched content from URL: {url} ({len(content)} chars)"
                    )
                    if self.valves.DEBUG_SEARCH:
                        logger.info(
                            "SEARCH DEBUG: fetch_content_ok url=%r length=%d",
                            url,
                            len(content),
                        )
                else:
                    logger.warning(f"Failed to fetch useful content from URL: {url}")
                    if self.valves.DEBUG_SEARCH:
                        logger.info(
                            "SEARCH DEBUG: fetch_content_failed url=%r length=%d",
                            url,
                            len(content) if content else 0,
                        )

            # If we still don't have useful content, mark as invalid
            if not snippet or len(snippet) < 200:
                if self.valves.DEBUG_SEARCH:
                    logger.info(
                        "SEARCH DEBUG: result_content_too_short url=%r length=%d",
                        url,
                        len(snippet) if snippet else 0,
                    )
                return {
                    "title": title or f"Result for '{query}'",
                    "url": url,
                    "content": snippet
                    or f"No substantial content available for this result.",
                    "query": query,
                    "valid": False,
                }

            # For repeated URLs, apply special sliding window treatment
            if repeat_count > 0:
                snippet = await self.handle_repeated_content(
                    snippet, url, query_embedding, repeat_count
                )

            # Calculate tokens in the content
            content_tokens = await self.count_tokens(snippet)

            # Get user preferences for PDV
            state = self.get_state()
            user_preferences = state.get("user_preferences", {})
            pdv = user_preferences.get("pdv")

            # Apply token limit if needed with adaptive scaling based on relevance
            max_tokens = await self.scale_token_limit_by_relevance(
                result, query_embedding, pdv
            )

            if content_tokens > max_tokens:
                # Process the content with token limiting using simple truncation with some padding
                try:
                    await self.emit_status(
                        "info", "Truncating content to token limit...", False
                    )

                    # Calculate character position based on token limit
                    char_ratio = max_tokens / content_tokens
                    char_limit = int(len(snippet) * char_ratio)

                    # Pad the limit to ensure we have complete sentences
                    padded_limit = min(len(snippet), int(char_limit * 1.1))

                    # Truncate content
                    truncated_content = snippet[:padded_limit]

                    # Find a good sentence break point
                    last_period = truncated_content.rfind(".")
                    if (
                        last_period > char_limit * 0.9
                    ):  # Only use period if it's near the target limit
                        truncated_content = truncated_content[: last_period + 1]

                    # If we got useful truncated content, use it
                    if truncated_content and len(truncated_content) > 100:
                        # Mark URL as actually selected (shown to user)
                        url_selected_count[url] = url_selected_count.get(url, 0) + 1
                        self.update_state("url_selected_count", url_selected_count)

                        # Store total tokens for this URL if not already done
                        if url not in url_token_counts:
                            url_token_counts[url] = content_tokens
                            self.update_state("url_token_counts", url_token_counts)

                        # Make sure this URL is in the master source table
                        if url not in master_source_table:
                            # (unchanged source table code)
                            source_type = "web"
                            if url.endswith(".pdf") or self.is_pdf_content:
                                source_type = "pdf"

                            # Try to get or create a good title
                            if not title or title == f"Result for '{query}'":
                                from urllib.parse import urlparse

                                parsed_url = urlparse(url)
                                if source_type == "pdf":
                                    file_name = parsed_url.path.split("/")[-1]
                                    title = (
                                        file_name.replace(".pdf", "")
                                        .replace("-", " ")
                                        .replace("_", " ")
                                    )
                                else:
                                    title = parsed_url.netloc

                            source_id = f"S{len(master_source_table) + 1}"
                            master_source_table[url] = {
                                "id": source_id,
                                "title": title,
                                "content_preview": truncated_content[:500],
                                "source_type": source_type,
                                "accessed_date": self.research_date,
                                "cited_in_sections": set(),
                            }
                            self.update_state(
                                "master_source_table", master_source_table
                            )

                            # Count tokens in truncated content
                            tokens = await self.count_tokens(truncated_content)

                            # Add timestamp to the result
                            result["timestamp"] = datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                        return {
                            "title": title,
                            "url": url,
                            "content": truncated_content,
                            "query": query,
                            "repeat_count": repeat_count,
                            "tokens": tokens,
                            "valid": True,
                        }
                except Exception as e:
                    logger.error(f"Error in token-based truncation: {e}")
                    # If truncation fails, we'll fall back to using original content with hard limit

            # If we haven't returned yet, use the original content with token limiting
            # Mark URL as actually selected (shown to user)
            url_selected_count[url] = url_selected_count.get(url, 0) + 1
            self.update_state("url_selected_count", url_selected_count)

            # Store total tokens for this URL if not already done
            if url not in url_token_counts:
                url_token_counts[url] = content_tokens
                self.update_state("url_token_counts", url_token_counts)

            # Make sure this URL is in the master source table
            if url not in master_source_table:
                source_type = "web"
                if url.endswith(".pdf") or self.is_pdf_content:
                    source_type = "pdf"

                # Try to get or create a good title
                if not title or title == f"Result for '{query}'":
                    from urllib.parse import urlparse

                    parsed_url = urlparse(url)
                    if source_type == "pdf":
                        file_name = parsed_url.path.split("/")[-1]
                        title = (
                            file_name.replace(".pdf", "")
                            .replace("-", " ")
                            .replace("_", " ")
                        )
                    else:
                        title = parsed_url.netloc

                source_id = f"S{len(master_source_table) + 1}"
                master_source_table[url] = {
                    "id": source_id,
                    "title": title,
                    "content_preview": snippet[:500],
                    "source_type": source_type,
                    "accessed_date": self.research_date,
                    "cited_in_sections": set(),
                }
                self.update_state("master_source_table", master_source_table)

            # If over token limit, truncate
            if content_tokens > max_tokens:
                # Estimate character position based on token limit
                char_ratio = max_tokens / content_tokens
                char_limit = int(len(snippet) * char_ratio)
                limited_content = snippet[:char_limit]
                # Actually count tokens rather than assuming max_tokens
                tokens = await self.count_tokens(limited_content)
            else:
                limited_content = snippet
                tokens = content_tokens

                # Add timestamp to the result
                result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return {
                "title": title,
                "url": url,
                "content": limited_content,
                "query": query,
                "repeat_count": repeat_count,
                "tokens": tokens,
                "valid": True,
            }

        except Exception as e:
            logger.error(f"Unhandled error in process_search_result: {e}")
            # Return a failure result
            error_msg = f"Error processing search result: {str(e)}\n\nOriginal snippet: {snippet[:1000] if snippet else 'No content available'}"
            tokens = await self.count_tokens(error_msg)

            return {
                "title": title or f"Error processing result for '{query}'",
                "url": url,
                "content": error_msg,
                "query": query,
                "repeat_count": repeat_count if "repeat_count" in locals() else 0,
                "tokens": tokens,
                "valid": False,
            }

    async def _try_openwebui_search(self, query: str) -> List[Dict]:
        """Delegate to SearchClient for OpenWebUI search."""
        return await self.search_client._try_openwebui_search(query)

    async def _fallback_search(self, query: str) -> List[Dict]:
        """Delegate to SearchClient for fallback search."""
        return await self.search_client._fallback_search(query)

    async def search_web(self, query: str) -> List[Dict]:
        """Perform web search via the SearchClient helper."""
        return await self.search_client.search_web(query)

    async def select_most_relevant_results(
        self,
        results: List[Dict],
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Select the most relevant results from extra results pool using semantic transformations with similarity caching"""
        if not results:
            return results

        # If we only have the base needed amount or fewer, return them all
        base_results_per_query = self.valves.SEARCH_RESULTS_PER_QUERY
        if len(results) <= base_results_per_query:
            return results

        # Get state for URL tracking
        state = self.get_state()
        url_selected_count = state.get("url_selected_count", {})

        # Count URLs that have been repeated at REPEATS_BEFORE_EXPANSION times or more
        repeat_count = 0
        for url, count in url_selected_count.items():
            if count >= self.valves.REPEATS_BEFORE_EXPANSION:
                repeat_count += 1

        # Calculate additional results to fetch based on repeat count
        additional_results = min(repeat_count, self.valves.EXTRA_RESULTS_PER_QUERY)
        results_to_select = base_results_per_query + additional_results

        # Calculate relevance scores for each result
        relevance_scores = []

        # Get transformation if available
        state = self.get_state()
        transformation = state.get("semantic_transformations")

        # Get similarity cache
        similarity_cache = state.get("similarity_cache", {})

        # Process domain priority valve value (if provided)
        priority_domains = []
        if hasattr(self.valves, "DOMAIN_PRIORITY") and self.valves.DOMAIN_PRIORITY:
            # Split by commas and/or spaces
            domain_input = self.valves.DOMAIN_PRIORITY
            # Replace commas with spaces, then split by spaces
            domain_items = domain_input.replace(",", " ").split()
            # Remove empty items and add to priority domains
            priority_domains = [
                item.strip().lower() for item in domain_items if item.strip()
            ]
            if priority_domains:
                logger.info(f"Using priority domains: {priority_domains}")

        # Process content priority valve value (if provided)
        priority_keywords = []
        if hasattr(self.valves, "CONTENT_PRIORITY") and self.valves.CONTENT_PRIORITY:
            # Split by commas and/or spaces, handling quoted phrases
            content_input = self.valves.CONTENT_PRIORITY

            # Function to parse keywords, respecting quotes
            def parse_keywords(text):
                keywords = []
                # Pattern for quoted phrases or words
                pattern = r"\'([^\']+)\'|\"([^\"]+)\"|(\S+)"

                matches = re.findall(pattern, text)
                for match in matches:
                    # Each match is a tuple with three groups (one will contain the text)
                    keyword = match[0] or match[1] or match[2]
                    if keyword:
                        keywords.append(keyword.lower())
                return keywords

            priority_keywords = parse_keywords(content_input)
            if priority_keywords:
                logger.info(f"Using priority keywords: {priority_keywords}")

        # Get multiplier values from valves or use defaults
        domain_multiplier = getattr(self.valves, "DOMAIN_MULTIPLIER", 1.5)
        keyword_multiplier_per_match = getattr(
            self.valves, "KEYWORD_MULTIPLIER_PER_MATCH", 1.1
        )
        max_keyword_multiplier = getattr(self.valves, "MAX_KEYWORD_MULTIPLIER", 2.0)

        for i, result in enumerate(results):
            try:
                # Get a snippet for evaluation
                snippet = result.get("snippet", "")
                url = result.get("url", "")

                # If snippet is too short and URL is available, fetch a bit of content
                if len(snippet) < self.valves.RELEVANCY_SNIPPET_LENGTH and url:
                    try:
                        await self.emit_status(
                            "info",
                            f"Fetching snippet for relevance check: {url[:50]}...",
                            False,
                        )
                        # Only fetch the first part of the content for evaluation
                        content_preview = await self.content_processor.fetch_content(url)
                        if content_preview:
                            snippet = content_preview[
                                : self.valves.RELEVANCY_SNIPPET_LENGTH
                            ]
                    except Exception as e:
                        logger.error(f"Error fetching content for relevance check: {e}")

                # Calculate relevance if we have enough content
                if snippet and len(snippet) > 100:
                    # FIRST, CHECK FOR VOCABULARY LIST
                    words = re.findall(r"\b\w+\b", snippet[:2000].lower())
                    if len(words) > 150:  # Only check if enough words
                        unique_words = set(words)
                        unique_ratio = len(unique_words) / len(words)
                        if (
                            unique_ratio > 0.98
                        ):  # Extremely high uniqueness = vocabulary list
                            logger.warning(
                                f"Skipping likely vocabulary list: {unique_ratio:.3f} uniqueness ratio"
                            )
                            # Assign a very low similarity score
                            similarity = 0.01
                            relevance_scores.append((i, similarity))
                            result["similarity"] = similarity
                            continue  # Skip the expensive embedding calculation

                    # Get embedding for the snippet
                    snippet_embedding = await self.get_embedding(snippet)

                    if snippet_embedding:
                        # Apply transformation to query only (Alternative A)
                        if transformation:
                            # Transform the query, not the content
                            transformed_query = (
                                await self.apply_semantic_transformation(
                                    query_embedding, transformation
                                )
                            )

                            # Calculate similarity between untransformed content and transformed query
                            similarity = cosine_similarity(
                                [snippet_embedding], [transformed_query]
                            )[0][0]
                        else:
                            # Calculate basic similarity if no transformation
                            similarity = cosine_similarity(
                                [snippet_embedding], [query_embedding]
                            )[0][0]

                        # Track original similarity for logging
                        original_similarity = similarity

                        # Apply domain multiplier if priority domains are set
                        if priority_domains and url:
                            url_lower = url.lower()
                            if any(domain in url_lower for domain in priority_domains):
                                similarity *= domain_multiplier
                                logger.debug(
                                    f"Applied domain multiplier {domain_multiplier}x to URL: {url}"
                                )

                        # Apply keyword multiplier if priority keywords are set
                        if priority_keywords and snippet:
                            snippet_lower = snippet.lower()
                            # Count matching keywords
                            keyword_matches = [
                                keyword
                                for keyword in priority_keywords
                                if keyword in snippet_lower
                            ]
                            keyword_count = len(keyword_matches)

                            if keyword_count > 0:
                                # Calculate cumulative multiplier (multiply by keyword_multiplier_per_match for each match)
                                # But cap at max_keyword_multiplier
                                cumulative_multiplier = min(
                                    max_keyword_multiplier,
                                    keyword_multiplier_per_match**keyword_count,
                                )
                                similarity *= cumulative_multiplier
                                logger.debug(
                                    f"Applied keyword multiplier {cumulative_multiplier:.2f}x "
                                    f"({keyword_count} keywords matched: {', '.join(keyword_matches[:3])}) to result {i}"
                                )

                        # Cap at 0.99 to avoid perfect scores
                        similarity = min(0.99, similarity)

                        # Log the full transformation if multipliers were applied
                        if similarity != original_similarity:
                            logger.info(
                                f"Result {i} multiplied: {original_similarity:.3f} → {similarity:.3f}"
                            )

                        # Store similarity in the result object for later use in topic dampening
                        result["similarity"] = similarity

                        # Apply penalty for repeated URLs
                        repeat_penalty = 1.0
                        url_repeats = url_selected_count.get(url, 0)
                        if url_repeats > 0:
                            # Apply a progressive penalty based on number of repeats
                            # More repeats = lower score (0.9, 0.8, 0.7, etc.)
                            repeat_penalty = max(0.5, 1.0 - (0.1 * url_repeats))
                            logger.debug(
                                f"Applied repeat penalty of {repeat_penalty} to URL: {url}"
                            )

                        # Apply penalty to similarity score
                        similarity *= repeat_penalty

                        # Store score for sorting
                        relevance_scores.append((i, similarity))

                        # Also store in the result for future use
                        result["similarity"] = similarity
                    else:
                        # No embedding, assign low score
                        relevance_scores.append((i, 0.1))
                        result["similarity"] = 0.1
                else:
                    # Insufficient content, assign low score
                    relevance_scores.append((i, 0.0))
                    result["similarity"] = 0.0

            except Exception as e:
                logger.error(f"Error calculating relevance for result {i}: {e}")
                relevance_scores.append((i, 0.0))
                result["similarity"] = 0.0

        # Update similarity cache
        self.update_state("similarity_cache", similarity_cache)

        # Sort by relevance score (highest first)
        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top results based on the dynamic count
        selected_indices = [x[0] for x in relevance_scores[:results_to_select]]
        selected_results = [results[i] for i in selected_indices]

        # Log selection information
        logger.info(
            f"Selected {len(selected_results)} most relevant results from {len(results)} total (added {additional_results} due to repeats)"
        )
        # Collect all content and quality factors first
        all_content = []
        for result in selected_results:
            content = result.get("content", "")[:2000]
            if content:
                # Use similarity as quality factor, normalize between 0.5-1.0
                quality = 0.5
                if "similarity" in result:
                    quality = 0.5 + (result["similarity"] * 0.5)
                all_content.append((content, quality))

        # Update ALL coverage in a single call
        if all_content:
            # Just grab dimensions once
            state = self.get_state()
            dims = state.get("research_dimensions")
            if dims and "coverage" in dims:
                coverage = np.array(dims["coverage"])

                # Process each content item sequentially
                for content, quality in all_content:
                    embed = await self.get_embedding(content[:2000])
                    if not embed:
                        continue
                    projection = np.dot(embed, np.array(dims["eigenvectors"]).T)
                    contribution = np.abs(projection) * quality

                    # Update coverage directly
                    for i in range(min(len(contribution), len(coverage))):
                        coverage[i] += contribution[i] * (1 - coverage[i] / 2)

                # Normalize once at the end
                coverage = np.minimum(coverage, 3.0) / 3.0

                # Save back
                dims["coverage"] = coverage.tolist()
                self.update_state("research_dimensions", dims)
                self.update_state("latest_dimension_coverage", coverage.tolist())

                # Log dimension updates for debugging
                state = self.get_state()
                research_dimensions = state.get("research_dimensions")
                if research_dimensions:
                    coverage = research_dimensions.get("coverage", [])
                    logger.debug(
                        f"Dimension coverage after result: {[round(c * 100) for c in coverage[:3]]}%..."
                    )

        return selected_results

    async def check_result_relevance(
        self,
        result: Dict,
        query: str,
        outline_items: Optional[List[str]] = None,
    ) -> bool:
        """Check if a search result is relevant to the query and research outline using a lightweight model"""
        if not self.valves.QUALITY_FILTER_ENABLED:
            return True  # Skip filtering if disabled

        # Get similarity score from result - access it correctly
        similarity = result.get("similarity", 0.0)

        # Skip filtering for very high similarity scores
        if similarity >= self.valves.QUALITY_SIMILARITY_THRESHOLD:
            logger.info(
                f"Result passed quality filter automatically with similarity {similarity:.3f}"
            )
            return True

        # Get content from the result
        content = result.get("content", "")
        title = result.get("title", "")
        url = result.get("url", "")

        if not content or len(content) < 200:
            logger.warning(
                f"Content too short for quality filtering, accepting by default"
            )
            return True

        # Create prompt for relevance checking
        relevance_prompt = {
            "role": "system",
            "content": """You are evaluating the relevance of a search result to a research query. 
Your task is to determine if the content is actually relevant to what the user is researching.

Answer with ONLY "Yes" if the content is relevant to the research query or "No" if it is:
- Not related to the core topic
- An advertisement disguised as content
- About a different product/concept with similar keywords
- So general or vague that it provides no substantive information
- Littered with HTML or CSS to the point of being unreadable

Reply with JUST "Yes" or "No" - no explanation or other text.""",
        }

        # Create context with query, outline, and full content
        context = f"Research Query: {query}\n\n"

        if outline_items and len(outline_items) > 0:
            context += "Research Outline Topics:\n"
            for item in outline_items[:5]:  # Limit to first 5 items
                context += f"- {item}\n"
            context += "\n"

        context += f"Result Title: {title}\n"
        context += f"Result URL: {url}\n\n"
        context += f"Content:\n{content}\n\n"
        context += f"""Is the above content relevant to this query: "{query}"? Answer with ONLY 'Yes' or 'No'."""

        try:
            # Use quality filter model
            quality_model = self.valves.QUALITY_FILTER_MODEL

            response = await self.generate_completion(
                quality_model,
                [relevance_prompt, {"role": "user", "content": context}],
                temperature=self.valves.TEMPERATURE
                * 0.2,  # Use your valve system with adjustment
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                answer = response["choices"][0]["message"]["content"].strip().lower()

                # Parse the response to get yes/no
                is_relevant = "yes" in answer.lower() and "no" not in answer.lower()

                logger.info(
                    f"Quality check for result: {'RELEVANT' if is_relevant else 'NOT RELEVANT'} (sim={similarity:.3f})"
                )

                return is_relevant
            else:
                logger.warning(
                    "Failed to get response from quality model, accepting by default"
                )
                return True

        except Exception as e:
            logger.error(f"Error in quality filtering: {e}")
            return True  # Accept by default on error
