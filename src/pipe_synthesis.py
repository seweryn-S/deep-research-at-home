from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeSynthesisMixin:
    async def generate_synthesis_outline(
        self,
        original_outline: List[Dict],
        completed_topics: Set[str],
        user_query: str,
        research_results: List[Dict],
    ) -> List[Dict]:
        """Generate a refined research outline for synthesis that better integrates additional research areas"""

        state = self.get_state()

        # Get the number of elapsed cycles
        elapsed_cycles = len(state.get("cycle_summaries", []))

        # Create a prompt for generating the synthesis outline
        synthesis_outline_prompt = {
            "role": "system",
            "content": f"""You are a post-graduate academic scholar reorganizing a research outline to be used in writing a comprehensive research report.
			
	Create a refined outline that condenses key topics/subtopics and insights from the current outline, and focuses on addressing the original query in areas best supported by the research.
    Aim to have approximately {round((elapsed_cycles * 0.25) + 2)} main topics and {round((elapsed_cycles * 0.8) + 5)} subtopics in your revised outline.

    The original user query was: "{user_query}".

    Your refined outline must:
	1. Appropriately incorporate relevant new topics discovered along the way that are directly relevant to the research "core" and original user query.
	2. Tailors the outline to reflect the progress and outcome of research activities without getting distracted by irrelevant results or specific examples, brands, locations, etc.
    3. Unite how research has evolved, and the reference material obtained during research, with the initial purpose and scope, prioritizing the initial purpose and scope.
    4. Where appropriate, reign in the representation of tangential research branches to refocus on topics more directly related to the original query.

    Your refined outline must NOT:
    1. Attempt to trump up, downplay, remove, soften, qualify, or otherwise modify the representation of research topics due to your own biases, preferences, or interests.
    2. Include main topics intended to serve as an introduction or conclusion for the full report.
    3. Focus on topics explored during research that don't actually serve to address the user's query or are fully tangent to it, or overly emphasize specific cases.
    4. Include any other text - please only respond with the outline. 
    
    The goal is to create a refined outline reflecting a logical narrative and informational flow for the final comprehensive report based on the user's query and gathered research.
	
	Format your response as a valid JSON object with the following structure:
	{{"outline": [
	  {{"topic": "Main topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]}},
	  {{"topic": "Main topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}}
	]}}""",
        }

        # Calculate similarity of research results to the research outline
        result_scores = []
        outline_text = "\n".join(
            [topic_item["topic"] for topic_item in original_outline]
        )

        # Check if we have a cached outline embedding
        state = self.get_state()
        outline_embedding_key = f"outline_embedding_{hash(outline_text)}"
        outline_embedding = state.get(outline_embedding_key)

        if not outline_embedding:
            outline_embedding = await self.get_embedding(outline_text)
            if outline_embedding:
                # Cache the outline embedding
                self.update_state(outline_embedding_key, outline_embedding)

        # Initialize outline_context
        outline_context = ""
        if outline_embedding:
            for i, result in enumerate(research_results):
                content = result.get("content", "")
                if not content:
                    continue

                # Check cache first for result embedding
                result_key = f"result_embedding_{hash(result.get('url', ''))}"
                content_embedding = state.get(result_key)

                if not content_embedding:
                    content_embedding = await self.get_embedding(content[:2000])
                    if content_embedding:
                        # Cache the result embedding
                        self.update_state(result_key, content_embedding)

                if content_embedding:
                    similarity = cosine_similarity(
                        [content_embedding], [outline_embedding]
                    )[0][0]
                    result_scores.append((i, similarity))

            # Sort results by similarity to outline in reverse order (most similar last)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_results = [research_results[i] for i, _ in result_scores]

            # Add sorted results to context
            outline_context += "\n### Research Results:\n\n"
            for result in sorted_results:
                outline_context += f"Title: {result.get('title', 'Untitled')}\n"
                outline_context += f"Content: {result.get('content', '')}\n\n"

        # Build context from the original outline and research results
        outline_context = "### Original Research Outline:\n\n"

        for topic_item in original_outline:
            outline_context += f"- {topic_item['topic']}\n"
            for subtopic in topic_item.get("subtopics", []):
                outline_context += f"  - {subtopic}\n"

        # Add semantic dimensions if available
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if research_dimensions:
            try:
                dimension_coverage = research_dimensions.get("coverage", [])

                # Create dimension labels for better context
                dimension_labels = await self.translate_dimensions_to_words(
                    research_dimensions, dimension_coverage
                )

                if dimension_coverage:
                    outline_context += "\n### Research Dimensions Coverage:\n"
                    for dim in dimension_labels[:10]:  # Limit to top 10 dimensions
                        outline_context += f"- {dim.get('words', 'Dimension ' + str(dim.get('dimension', 0)))}:  {dim.get('coverage', 0)}% covered\n"

            except Exception as e:
                logger.error(
                    f"Error adding research dimensions to outline context: {e}"
                )

        # Create messages for the model
        messages = [
            synthesis_outline_prompt,
            {
                "role": "user",
                "content": f"{outline_context}\n\nGenerate a refined research outline following the instructions and format in the system prompt.",
            },
        ]

        # Generate the synthesis outline
        try:
            await self.emit_status(
                "info", "Generating refined outline for synthesis...", False
            )

            # Use synthesis model for this task
            synthesis_model = self.get_synthesis_model()
            response = await self.generate_completion(
                synthesis_model, messages, temperature=self.valves.SYNTHESIS_TEMPERATURE
            )
            outline_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                # First try standard JSON extraction
                json_start = outline_content.find("{")
                json_end = outline_content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    outline_json_str = outline_content[json_start:json_end]
                    try:
                        outline_data = json.loads(outline_json_str)
                        synthesis_outline = outline_data.get("outline", [])
                        if synthesis_outline:
                            return synthesis_outline
                    except (json.JSONDecodeError, ValueError):
                        # If standard approach fails, try regex approach
                        pass

                # Use regex to find any JSON structure containing "outline" array
                import re

                json_pattern = r'(\{[^{}]*"outline"\s*:\s*\[[^\[\]]*\][^{}]*\})'
                matches = re.findall(json_pattern, outline_content, re.DOTALL)

                for match in matches:
                    try:
                        outline_data = json.loads(match)
                        synthesis_outline = outline_data.get("outline", [])
                        if synthesis_outline:
                            return synthesis_outline
                    except:
                        continue

                # If no valid JSON found, try a more aggressive repair approach
                # Look for anything that resembles the outline structure
                topic_pattern = (
                    r'"topic"\s*:\s*"([^"]*)"\s*,\s*"subtopics"\s*:\s*\[(.*?)\]'
                )
                topics_matches = re.findall(topic_pattern, outline_content, re.DOTALL)

                if topics_matches:
                    synthetic_outline = []
                    for topic_match in topics_matches:
                        topic = topic_match[0]
                        subtopics_str = topic_match[1]
                        # Extract subtopics strings - look for quoted strings
                        subtopics = re.findall(r'"([^"]*)"', subtopics_str)
                        synthetic_outline.append(
                            {"topic": topic, "subtopics": subtopics}
                        )

                    if synthetic_outline:
                        return synthetic_outline

                # All extraction methods failed, return original outline
                return original_outline

            except Exception as e:
                logger.error(f"Error parsing synthesis outline JSON: {e}")
                return original_outline

        except Exception as e:
            logger.error(f"Error generating synthesis outline: {e}")
            return original_outline

    async def generate_subtopic_content_with_citations(
        self,
        section_title: str,
        subtopic: str,
        original_query: str,
        research_results: List[Dict],
        synthesis_model: str,
        is_follow_up: bool = False,
        previous_summary: str = "",
    ) -> Dict:
        """Generate content for a single subtopic with numbered citations"""
        # Only emit status if we haven't seen this section yet
        if not hasattr(self, "_seen_subtopics"):
            self._seen_subtopics = set()

        # Only emit status if we haven't seen this subtopic yet
        if subtopic not in self._seen_subtopics:
            await self.emit_status(
                "info", f"Generating content for subtopic: {subtopic}...", False
            )
            self._seen_subtopics.add(subtopic)

        # Get state
        state = self.get_state()

        # Get relevance cache or initialize it
        relevance_cache = state.get("subtopic_relevance_cache", {})

        # Create embedding cache keys for efficiency
        query_embedding_key = f"query_embedding_{hash(original_query)}"
        subtopic_embedding_key = f"subtopic_embedding_{hash(subtopic)}"
        combined_embedding_key = (
            f"combined_embedding_{hash(original_query)}_{hash(subtopic)}"
        )

        # Create a prompt specific to this subtopic
        subtopic_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research assistant writing a concise subsection (1-3 paragraphs) about "{subtopic}" 
        for a comprehensive combined research report addressing this query: "{original_query}" based on internet research results.
        
        Your subsection MUST:
        1. Focus specifically on the subtopic "{subtopic}" within the broader section "{section_title}".
        2. Make FULL use of the provided research sources, and ONLY the provided sources.
        3. Include IN-TEXT CITATIONS for all information from sources, using ONLY the numerical IDs provided in the source list, e.g. [1], [4], etc.
        4. Follow a structure that best fits the subtopic subject matter. Aim for an academic report style while tolerating flexibility as appropriate.
        5. Only be written on the subtopic matter - consider how your subsection will be combined with others in a greater research report.
        6. Be written to a length, between 1 medium paragraph and 3 long paragraphs, based on the subtopic's perceived importance to the research query.
        
        Your subsection must NOT:
        1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication. 
        2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
        3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
        4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
        5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
        6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.
        7. Include any explicit "Sources", "References", or "Bibliography" section or list of URLs at the end of the subsection. All referencing must remain inline via numerical citations only.
    
        You must accurately cite your sources to avoid plagiarizing. Citations MUST be numerical and correspond to the correct source ID in the provided list.
        Do not combine multiple IDs in one citation tag. Do NOT add a separate source list or bibliography at the end of the subsection. Please respond with just the subsection body, no intro or title.""",
        }

        # Create a combined embedding for query + subtopic with state-level caching
        combined_embedding = state.get(combined_embedding_key)

        if combined_embedding is None:
            # Check if we already have the individual embeddings cached in state
            query_embedding = state.get(query_embedding_key)
            subtopic_embedding = state.get(subtopic_embedding_key)

            # Get query embedding if not already cached
            if query_embedding is None:
                try:
                    query_embedding = await self.get_embedding(original_query)
                    if query_embedding:
                        # Cache at state level to avoid repeated API calls
                        self.update_state(query_embedding_key, query_embedding)
                except Exception as e:
                    logger.error(f"Error getting query embedding: {e}")

            # Get subtopic embedding if not already cached
            if subtopic_embedding is None:
                try:
                    subtopic_embedding = await self.get_embedding(subtopic)
                    if subtopic_embedding:
                        # Cache at state level to avoid repeated API calls
                        self.update_state(subtopic_embedding_key, subtopic_embedding)
                except Exception as e:
                    logger.error(f"Error getting subtopic embedding: {e}")

            # Create combined embedding if both components exist
            if query_embedding and subtopic_embedding:
                try:
                    # Combine with equal weight
                    combined_array = (
                        np.array(query_embedding) * 0.5
                        + np.array(subtopic_embedding) * 0.5
                    )
                    # Normalize
                    norm = np.linalg.norm(combined_array)
                    if norm > 1e-10:
                        combined_array = combined_array / norm
                    combined_embedding = combined_array.tolist()
                    # Cache the combined embedding
                    self.update_state(combined_embedding_key, combined_embedding)
                except Exception as e:
                    logger.error(f"Error creating combined embedding: {e}")

        # If combined embedding failed or doesn't exist, fall back to subtopic embedding
        if not combined_embedding:
            # Check if we have cached subtopic embedding
            subtopic_embedding = state.get(subtopic_embedding_key)
            if not subtopic_embedding:
                # Try to get it now
                subtopic_embedding = await self.get_embedding(subtopic)
                # Cache if successful
                if subtopic_embedding:
                    self.update_state(subtopic_embedding_key, subtopic_embedding)

            combined_embedding = subtopic_embedding

        # Build context from research results that might be relevant to this subtopic
        subtopic_context = f"# Subtopic to Write: {subtopic}\n"
        subtopic_context += f"# Within Section: {section_title}\n\n"

        # Add the research outline for context
        subtopic_context += "## Research Outline Context:\n"
        state = self.get_state()
        research_state = state.get("research_state") or {}
        synthesis_outline = research_state.get("research_outline", [])
        if synthesis_outline:
            for topic_item in synthesis_outline:
                topic = topic_item.get("topic", "")
                if topic == section_title:
                    subtopic_context += f"**Current Section: {topic}**\n"
                else:
                    subtopic_context += f"Section: {topic}\n"

                for st in topic_item.get("subtopics", []):
                    if st == subtopic:
                        subtopic_context += f"  - **Current Subtopic: {st}**\n"
                    else:
                        subtopic_context += f"  - {st}\n"
            subtopic_context += "\n"

        # Create a unique cache key for this subtopic
        subtopic_key = f"{section_title}_{subtopic}"

        # Calculate relevance scores for each result
        subtopic_results = []
        result_scores = []

        # Check if we have cached relevance scores for this subtopic
        if subtopic_key in relevance_cache:
            logger.info(f"Using cached relevance scores for subtopic: {subtopic}")
            result_scores = relevance_cache[subtopic_key]
            # Sort by relevance score (highest first)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            # Map back to research results
            subtopic_results = [
                research_results[i]
                for i, _ in result_scores
                if i < len(research_results)
            ]
        elif combined_embedding:
            # Calculate relevance scores using combined query+subtopic embedding
            for i, result in enumerate(research_results):
                content = result.get("content", "")
                if not content:
                    continue

                # Create a cache key for this result's embedding
                result_key = f"result_{hash(result.get('url', ''))}"
                content_embedding = state.get(result_key)

                if not content_embedding:
                    content_embedding = await self.get_embedding(content[:2000])
                    # Cache the content embedding if valid
                    if content_embedding:
                        self.update_state(result_key, content_embedding)

                if content_embedding:
                    similarity = cosine_similarity(
                        [content_embedding], [combined_embedding]
                    )[0][0]
                    result_scores.append((i, similarity))

            # Cache the relevance scores for this subtopic
            relevance_cache[subtopic_key] = result_scores
            self.update_state("subtopic_relevance_cache", relevance_cache)

            # Sort by relevance score (highest first)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            # Map to research results
            subtopic_results = [research_results[i] for i, _ in result_scores]
        else:
            # If no embedding, just use all results
            subtopic_results = research_results

        # Calculate how many results to include based on number of cycles and vibes
        top_results_count = max(
            3, min(len(subtopic_results), math.ceil(0.5 * self.valves.MAX_CYCLES + 3))
        )
        top_results = subtopic_results[:top_results_count]

        # Create source list with assigned IDs
        sources_for_subtopic = {}
        source_id = 1

        # Extract URLs and titles from top results, sort alphabetically by title
        sorted_results = sorted(top_results, key=lambda x: x.get("title", "").lower())

        for result in sorted_results:
            url = result.get("url", "")
            title = result.get("title", "Untitled Source")

            if url and url not in sources_for_subtopic:
                sources_for_subtopic[url] = {
                    "id": source_id,
                    "title": title,
                    "url": url,
                    "subtopic": subtopic,
                    "section": section_title,
                }
                source_id += 1

        # Add source list to context (at the beginning)
        subtopic_context += (
            "## Available Source List (Use ONLY these numerical citations):\n\n"
        )
        for url, source_data in sorted(
            sources_for_subtopic.items(), key=lambda x: x[1]["title"]
        ):
            subtopic_context += (
                f"[{source_data['id']}] {source_data['title']} - {url}\n"
            )

        subtopic_context += "\n## Research Results:\n\n"

        # Reorder results to have most relevant last (most recent in context)
        top_results.reverse()

        # Add the top results to context (most relevant last)
        for result in top_results:
            url = result.get("url", "")
            title = result.get("title", "Untitled Source")
            content = result.get("content", "")

            # Skip results without content
            if not content:
                continue

            # Get the source ID for this URL
            source_id = sources_for_subtopic.get(url, {}).get("id", "?")

            subtopic_context += f"Source ID: [{source_id}] {title}\n"
            subtopic_context += f"Content: {content}\n\n"

        # Include previous summary if this is a follow-up
        if is_follow_up and previous_summary:
            subtopic_context += "## Previous Research Summary:\n\n"
            subtopic_context += f"{previous_summary}...\n\n"

        # Prepare final instruction
        subtopic_context += f"""Using the provided research sources and referencing them with numerical citations [#], write a concise subsection about "{subtopic}" per the system prompt."""
        subtopic_context += f"""Every citation MUST be numerical (e.g., [1], [2]) corresponding to the source list provided."""
        subtopic_context += f"""Please use proper Markdown and write 1-3 focused paragraphs exclusively on this specific subtopic."""

        # Create messages array for completion
        messages = [subtopic_prompt, {"role": "user", "content": subtopic_context}]

        # Generate subtopic content
        try:
            # Calculate scaled temperature from the synthesis temperature valve
            scaled_temperature = (
                self.valves.TEMPERATURE
            )  # Use research model temperature for subtopics

            # Use research model for generating subtopics
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=scaled_temperature,
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                subtopic_content = response["choices"][0]["message"]["content"]

                # Count tokens in the subtopic content
                tokens = await self.count_tokens(subtopic_content)

                # Store content for later use
                subtopic_synthesized_content = state.get(
                    "subtopic_synthesized_content", {}
                )
                subtopic_synthesized_content[subtopic] = subtopic_content
                self.update_state(
                    "subtopic_synthesized_content", subtopic_synthesized_content
                )

                # Store source mapping for this subtopic
                subtopic_sources = state.get("subtopic_sources", {})
                subtopic_sources[subtopic] = sources_for_subtopic
                self.update_state("subtopic_sources", subtopic_sources)

                # Identify citations in this subtopic content
                subtopic_citations = []
                for url, source_data in sources_for_subtopic.items():
                    local_id = source_data.get("id")
                    if local_id is not None:
                        # Find all instances of this citation in the text
                        pattern = (
                            r"([^.!?]*(?:\["
                            + str(local_id)
                            + r"\]|&#"
                            + str(local_id)
                            + r")[^.!?]*[.!?])"
                        )
                        context_matches = re.findall(pattern, subtopic_content)

                        for match in context_matches:
                            citation = {
                                "marker": str(local_id),
                                "raw_text": f"[{local_id}]",
                                "text": match,
                                "url": url,
                                "section": section_title,
                                "subtopic": subtopic,
                                "suggested_title": source_data.get("title", ""),
                            }
                            subtopic_citations.append(citation)

                # Log the sources used
                logger.info(
                    f"Subtopic '{subtopic}' uses {len(sources_for_subtopic)} sources"
                )

                return {
                    "content": subtopic_content,  # Return original content with local IDs
                    "tokens": tokens,
                    "sources": sources_for_subtopic,
                    "citations": subtopic_citations,
                    "verified_citations": [],  # Verification happens later
                    "flagged_citations": [],  # Flagging happens later
                }
            else:
                return {
                    "content": f"*Error generating content for subtopic: {subtopic}*",
                    "tokens": 0,
                    "sources": {},
                    "citations": [],
                    "verified_citations": [],
                    "flagged_citations": [],
                }

        except Exception as e:
            logger.error(f"Error generating subtopic content for '{subtopic}': {e}")
            return {
                "content": f"*Error generating content for subtopic: {subtopic}*",
                "tokens": 0,
                "sources": {},
                "citations": [],
                "verified_citations": [],
                "flagged_citations": [],
            }

    async def generate_section_content_with_citations(
        self,
        section_title: str,
        subtopics: List[str],
        original_query: str,
        research_results: List[Dict],
        synthesis_model: str,
        is_follow_up: bool = False,
        previous_summary: str = "",
    ) -> Dict:
        """Generate content for a section by combining subtopics with citations"""
        # Use a static set to track which sections we've displayed status for
        if not hasattr(self, "_seen_sections"):
            self._seen_sections = set()

        # Only emit status if we haven't seen this section yet
        if section_title not in self._seen_sections:
            await self.emit_status(
                "info", f"Generating content for section: {section_title}...", False
            )
            self._seen_sections.add(section_title)

        # Get state
        state = self.get_state()

        # Generate content for each subtopic independently
        subtopic_contents = {}
        section_sources = {}
        all_section_citations = []
        total_tokens = 0

        for subtopic in subtopics:
            subtopic_result = await self.generate_subtopic_content_with_citations(
                section_title,
                subtopic,
                original_query,
                research_results,
                synthesis_model,
                is_follow_up,
                previous_summary if is_follow_up else "",
            )

            subtopic_contents[subtopic] = subtopic_result["content"]
            total_tokens += subtopic_result.get("tokens", 0)

            # Collect all citations from this subtopic
            if "citations" in subtopic_result:
                all_section_citations.extend(subtopic_result["citations"])

            # Merge sources with section sources, maintaining unique source IDs and tracking originals
            for url, source_data in subtopic_result["sources"].items():
                if url not in section_sources:
                    section_sources[url] = (
                        source_data.copy()
                    )  # Use copy to avoid reference issues

                # Store the original local ID with subtopic context for precise replacement
                # Add this information to the source data record
                if "original_ids" not in section_sources[url]:
                    section_sources[url]["original_ids"] = {}

                # Track which local ID was used in which subtopic
                local_id = source_data.get("id")
                if local_id is not None:
                    section_sources[url]["original_ids"][subtopic] = local_id

        # Build or update the global citation map with sources from this section
        master_source_table = state.get("master_source_table", {})
        global_citation_map = state.get("global_citation_map", {})

        # Add all section sources to global map if not already present
        for url, source_data in section_sources.items():
            if url not in global_citation_map:
                global_citation_map[url] = len(global_citation_map) + 1

            # Also add to master source table if not already there
            if url not in master_source_table:
                source_id = f"S{len(master_source_table) + 1}"
                master_source_table[url] = {
                    "id": source_id,
                    "title": source_data.get("title", "Untitled Source"),
                    "content_preview": "",
                    "source_type": "web" if not url.endswith(".pdf") else "pdf",
                    "accessed_date": self.research_date,
                    "cited_in_sections": set([section_title]),
                }
            elif section_title not in master_source_table[url].get(
                "cited_in_sections", set()
            ):
                # Update sections where this source is cited
                master_source_table[url]["cited_in_sections"].add(section_title)

        # Update state
        self.update_state("global_citation_map", global_citation_map)
        self.update_state("master_source_table", master_source_table)

        # Verify citations if enabled
        verified_citations = []
        flagged_citations = []

        if self.valves.VERIFY_CITATIONS and all_section_citations:
            # Group citations by URL for efficient verification
            citations_by_url = {}
            for citation in all_section_citations:
                url = citation.get("url")
                if url:
                    if url not in citations_by_url:
                        citations_by_url[url] = []
                    citations_by_url[url].append(citation)

            # Verify each URL's citations
            for url, citations in citations_by_url.items():
                try:
                    # Get source content
                    url_results_cache = state.get("url_results_cache", {})

                    # Check cache first
                    source_content = None
                    if url in url_results_cache:
                        source_content = url_results_cache[url]

                    # If not in cache, fetch source content
                    if not source_content or len(source_content) < 200:
                        source_content = await self.content_processor.fetch_content(url)

                    if source_content and len(source_content) >= 200:
                        # Add global ID to each citation for verification tracking
                        if url in global_citation_map:
                            global_id = global_citation_map[url]
                            for citation in citations:
                                citation["global_id"] = global_id

                        # Verify citations against source content
                        verification_results = await self.verify_citation_batch(
                            url, citations, source_content
                        )

                        # Sort verified and flagged citations
                        for result in verification_results:
                            if result.get("verified", False):
                                verified_citations.append(result)
                            elif result.get("flagged", False):
                                flagged_citations.append(result)
                    else:
                        # Mark as unverified but not flagged
                        for citation in citations:
                            citation["verified"] = False
                            citation["flagged"] = False
                except Exception as e:
                    logger.error(f"Error verifying citations for URL {url}: {e}")
                    # Mark as unverified but not flagged
                    for citation in citations:
                        citation["verified"] = False
                        citation["flagged"] = False

        # Now process each subtopic content to:
        # 1. Apply strikethrough to flagged citations
        # 2. Replace local citation IDs with global IDs
        processed_subtopic_contents = {}

        for subtopic, content in subtopic_contents.items():
            processed_content = content

            # Apply strikethrough to flagged citations
            flagged_sentences_for_subtopic = set()
            for citation in flagged_citations:
                if citation.get("subtopic") == subtopic and citation.get("text"):
                    flagged_sentences_for_subtopic.add(citation.get("text"))

            if flagged_sentences_for_subtopic:
                # Split content into sentences
                sentences = re.split(r"(?<=[.!?])\s+", processed_content)
                modified_sentences = []

                for sentence in sentences:
                    modified_sentence = sentence

                    # Check if this is a flagged sentence
                    for flagged_text in flagged_sentences_for_subtopic:
                        if flagged_text in sentence:
                            # Apply strikethrough
                            modified_sentence = f"~~{modified_sentence}~~"
                            break

                    modified_sentences.append(modified_sentence)

                # Join sentences back together
                processed_content = " ".join(modified_sentences)

                # Track applied strikethroughs
                citation_fixes = state.get("citation_fixes", [])
                for flagged_text in flagged_sentences_for_subtopic:
                    citation_fixes.append(
                        {
                            "section": section_title,
                            "subtopic": subtopic,
                            "reason": "Citation could not be verified",
                            "original_text": flagged_text,
                        }
                    )
                self.update_state("citation_fixes", citation_fixes)

            # Now replace all local citation IDs with global IDs using context-aware replacement
            # First handle single citations - standard pattern [n]
            for url, source_data in section_sources.items():
                # Check if this URL has a local ID for this specific subtopic
                original_ids = source_data.get("original_ids", {})
                local_id = original_ids.get(subtopic)

                if local_id is not None and url in global_citation_map:
                    global_id = global_citation_map[url]

                    # Replace local citation ID with global ID
                    pattern = r"\[" + re.escape(str(local_id)) + r"\]"
                    processed_content = re.sub(
                        pattern, f"[{global_id}]", processed_content
                    )

            # Now handle combined citations like [1, 2] or [1,2]
            # First, extract all citation groups from content
            combined_citation_pattern = r"\[(\d+(?:\s*,\s*\d+)+)\]"
            combined_matches = re.finditer(combined_citation_pattern, processed_content)

            # Process each combined citation group
            for match in combined_matches:
                original_citation = match.group(
                    0
                )  # The full citation group e.g. "[1, 2, 3]"
                citation_ids = match.group(1)  # Just the IDs part e.g. "1, 2, 3"

                # Extract individual IDs (handles both [1,2] and [1, 2] formats)
                local_ids = [
                    int(id_str.strip()) for id_str in re.split(r"\s*,\s*", citation_ids)
                ]

                # Convert each local ID to its global ID
                global_ids = []
                for local_id in local_ids:
                    # Find the URL(s) that had this local ID in this subtopic
                    for url, source_data in section_sources.items():
                        original_ids = source_data.get("original_ids", {})
                        if (
                            subtopic in original_ids
                            and original_ids[subtopic] == local_id
                        ):
                            if url in global_citation_map:
                                global_ids.append(str(global_citation_map[url]))

                # If we found global IDs, create the replacement citation
                if global_ids:
                    global_citation = f"[{', '.join(global_ids)}]"
                    # Replace just this specific citation instance
                    processed_content = processed_content.replace(
                        original_citation, global_citation, 1
                    )

            processed_subtopic_contents[subtopic] = processed_content

        # Combine subtopic contents into a section draft
        combined_content = ""
        for subtopic, content in processed_subtopic_contents.items():
            # Add subtopic heading
            combined_content += f"\n\n### {subtopic}\n\n"
            combined_content += f"{content}\n\n"

        # Only do smoothing if we have multiple subtopics
        if len(subtopics) > 1:
            # Review and smooth transitions between subtopics
            section_content = await self.smooth_section_transitions(
                section_title,
                subtopics,
                combined_content,
                original_query,
                synthesis_model,
            )
        else:
            section_content = combined_content

        # Track token counts
        memory_stats = self._ensure_memory_stats()
        section_tokens = memory_stats.get("section_tokens", {})
        section_tokens[section_title] = total_tokens
        memory_stats["section_tokens"] = section_tokens
        self.update_state("memory_stats", memory_stats)

        # Store content for later use
        section_synthesized_content = state.get("section_synthesized_content", {})
        section_synthesized_content[section_title] = section_content
        self.update_state("section_synthesized_content", section_synthesized_content)

        # Store section sources for later citation correlation
        section_sources_map = state.get("section_sources_map", {})
        section_sources_map[section_title] = section_sources
        self.update_state("section_sources_map", section_sources_map)

        # Store all citations for this section
        section_citations = state.get("section_citations", {})
        section_citations[section_title] = all_section_citations
        self.update_state("section_citations", section_citations)

        # Show section completion status
        await self.emit_status(
            "info",
            f"Section generated: {section_title}",
            False,
        )

        return {
            "content": section_content,
            "tokens": total_tokens,
            "sources": section_sources,
            "citations": all_section_citations,
            "verified_citations": verified_citations,
            "flagged_citations": flagged_citations,
        }

    async def smooth_section_transitions(
        self,
        section_title: str,
        subtopics: List[str],
        combined_content: str,
        original_query: str,
        synthesis_model: str,
    ) -> str:
        """Review and smooth transitions between subtopics in a section"""

        # Create a prompt for smoothing transitions
        smoothing_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research editor editing a section that combines multiple subtopics.
            
    Review the section content and improve it by:
    1. Restructuring subtopic content and makeup to better fit the greater context of the section and full report
    2. Ensuring consistent style and tone throughout the section and ensuring consistent use of proper Markdown
    3. Maintaining the exact factual content in sentences with numerical citations [#]
    4. Removing duplicate subtopic headings
    5. Moving sentences or concepts between subsections as appropriate and revising subsection headers to fit the content
    6. Removing any meta-commentary, e.g. "Okay, here's the section" or "I wrote the section while considering..."
    7. Making the section read as though it were written by one person with a cohesive strategy for assembling the section
    
    DO NOT:
    1. Remove, change, or edit ANY in-text citations or applied strikethrough
    2. Alter, censor, re-analyze, or edit the factual content in ANY way
    3. Add new information or qualifiers not present in the original
    4. Decouple the factual content of a sentence from its specific citation
    5. Include any introduction, conclusion, main title header, or meta-commentary - please return the section as requested with no other text
    6. Combine sentences containing in-text citations and/or strikethrough

    It is vitally important that your edits preserve the direct connection between any sentence and its in-text citation and/or applied strikethrough.
    You may relocate or lightly edit sentences with in-text citations or strikethrough if appropriate, as long as they maintain these features.""",
        }

        # Create context with the combined subtopics
        smoothing_context = f"# Section to Improve: '{section_title}'\n\n"
        smoothing_context += (
            f"This section is part of a research paper on: '{original_query}'\n\n"
        )

        # Add the research outline for better context
        state = self.get_state()
        research_state = state.get("research_state") or {}
        research_outline = research_state.get("research_outline", [])
        if research_outline:
            smoothing_context += f"## Full Research Outline:\n"
            for topic_item in research_outline:
                topic = topic_item.get("topic", "")
                if topic == section_title:
                    smoothing_context += f"**Current Section: {topic}**\n"
                else:
                    smoothing_context += f"Section: {topic}\n"

                for st in topic_item.get("subtopics", []):
                    smoothing_context += f"  - {st}\n"
            smoothing_context += "\n"

        smoothing_context += f"## Subtopics in this section:\n"
        for subtopic in subtopics:
            smoothing_context += f"- '{subtopic}'\n"

        smoothing_context += f"\n## Combined Section Content:\n\n{combined_content}\n\n"
        smoothing_context += f"Please improve this section by ensuring smooth transitions between subtopics while preserving all factual content and numerical citations."

        # Create messages for completion
        messages = [smoothing_prompt, {"role": "user", "content": smoothing_context}]

        try:
            # Use synthesis model for smoothing
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=self.valves.SYNTHESIS_TEMPERATURE
                * 0.7,  # Lower temperature for editing
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                improved_content = response["choices"][0]["message"]["content"]
                return improved_content
            else:
                # Return original if synthesis fails
                return combined_content

        except Exception as e:
            logger.error(
                f"Error smoothing transitions for section '{section_title}': {e}"
            )
            # Return original content on error
            return combined_content

    async def generate_bibliography(
        self, master_source_table, global_citation_map, compiled_sections=None
    ):
        """Generate a bibliography using sequential numbering based on actual citations in the report.

        compiled_sections:
            Optional mapping {section_title: text}. If omitted, falls back to
            the latest section_synthesized_content stored in state.
        """
        if not master_source_table:
            return {
                "bibliography": [],
                "title_to_global_id": {},
                "url_to_global_id": {},
            }

        # First, scan all compiled sections to find actually cited sources
        if compiled_sections is None:
            state = self.get_state()
            compiled_sections = state.get("section_synthesized_content", {})

        # Extract all citation numbers from the compiled text
        cited_numbers = set()
        for section_content in compiled_sections.values():
            # Find all citations in format [n] where n is a number
            citation_matches = re.findall(r"\[(\d+)\]", section_content)
            for num in citation_matches:
                cited_numbers.add(int(num))

        # Filter global_citation_map to only include cited sources
        cited_urls = {}
        for url, id_num in global_citation_map.items():
            if id_num in cited_numbers:
                cited_urls[url] = id_num

        # Sort URLs by their assigned citation ID
        sorted_urls = sorted(cited_urls.items(), key=lambda x: x[1])

        # Create bibliography entries based on cited sources only
        bibliography = []
        url_to_global_id = {}
        title_to_global_id = {}

        # Use the sequential numbers already assigned in global_citation_map
        for url, global_id in sorted_urls:
            # Get source data from master_source_table if available
            if url in master_source_table:
                source_data = master_source_table[url]
                title = source_data.get("title", "Untitled Source")
            else:
                logger.warning(
                    f"URL {url} in global_citation_map not found in master_source_table"
                )
                title = f"Source {global_id}"

            # Add bibliography entry using the actual correlated URL
            bibliography.append(
                {
                    "id": global_id,
                    "title": title,
                    "url": url,
                }
            )

            # Create mappings
            url_to_global_id[url] = global_id
            title_to_global_id[title] = global_id

        # Sort bibliography by citation ID
        bibliography.sort(key=lambda x: x["id"])

        logger.info(
            f"Generated bibliography with {len(bibliography)} cited entries (from {len(global_citation_map)} total sources)"
        )
        return {
            "bibliography": bibliography,
            "title_to_global_id": title_to_global_id,
            "url_to_global_id": url_to_global_id,
        }

    async def format_bibliography_list(self, bibliography):
        """Format the bibliography as Markdown footnotes.

        Each entry is rendered as:
            [^n]: Title. [url](url)
        so that in-text citations like [^n] are resolved correctly by Markdown renderers.
        """
        if not bibliography:
            return "No sources were referenced in this research."

        # Create bibliography header
        bib_list = "\n\n## Bibliography\n\n"

        # Add each bibliography entry
        for entry in bibliography:
            citation_id = entry["id"]
            title = entry["title"]
            url = entry["url"]

            # Format URL for markdown linking
            if url.startswith("http"):
                url_formatted = f"[{url}]({url})"
            else:
                url_formatted = url

            # Use standard Markdown footnote definition, e.g.:
            # [^1]: Title. [url](url)
            bib_list += f"[^{citation_id}]: {title}. {url_formatted}\n\n"

        return bib_list

    async def export_research_data(self) -> Dict:
        """Export the full research data including results, queries, timestamps, URLs, and content"""
        import os
        import json
        from datetime import datetime

        state = self.get_state()
        results_history = state.get("results_history", [])

        # Get current date and time for the export timestamp
        export_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare the export data structure
        research_state = state.get("research_state") or {}

        export_data = {
            "export_timestamp": export_timestamp,
            "research_date": self.research_date,
            "original_query": research_state.get("user_message", "Unknown query"),
            "results_count": len(results_history),
            "results": [],
        }

        # Process each result to include in the export
        for i, result in enumerate(results_history):
            # Add timestamp to result if not already present
            if "timestamp" not in result:
                # As a fallback, create a synthetic timestamp based on position in history
                from datetime import timedelta

                synthetic_time = datetime.now() - timedelta(
                    minutes=(len(results_history) - i)
                )
                result_timestamp = synthetic_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                result_timestamp = result.get("timestamp")

            # Format the result for export
            export_result = {
                "index": i + 1,
                "timestamp": result_timestamp,
                "query": result.get("query", "Unknown query"),
                "url": result.get("url", ""),
                "title": result.get("title", "Untitled"),
                "tokens": result.get("tokens", 0),
                "content": result.get("content", ""),
                "similarity": result.get("similarity", 0.0),
            }

            export_data["results"].append(export_result)

        # Generate filename based on the research query (sanitized) and timestamp
        query_text = research_state.get("user_message", "research")
        # Sanitize the query for a filename (first 30 chars, remove unsafe chars)
        query_for_filename = (
            "".join(c if c.isalnum() or c in " -_" else "_" for c in query_text[:30])
            .strip()
            .replace(" ", "_")
        )
        filename = f"research_export_{query_for_filename}_{file_timestamp}.txt"

        # Determine file path
        # Use OpenWebUI directory (or fallback to current directory)
        try:
            from open_webui import get_app_dir

            export_dir = get_app_dir()
        except:
            export_dir = os.getcwd()  # Fallback to current directory

        filepath = os.path.join(export_dir, filename)

        # Ensure the directory exists
        os.makedirs(export_dir, exist_ok=True)

        # Write export data to file
        with open(filepath, "w", encoding="utf-8") as f:
            # Write a human-readable header
            f.write(f"# Research Export: {export_data['original_query']}\n")
            f.write(f"# Date: {export_data['export_timestamp']}\n")
            f.write(f"# Results: {export_data['results_count']}\n\n")

            # Write each result with clear separation
            for result in export_data["results"]:
                f.write(f"=== RESULT {result['index']} ===\n")
                f.write(f"Timestamp: {result['timestamp']}\n")
                f.write(f"Query: {result['query']}\n")
                f.write(f"URL: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Tokens: {result['tokens']}\n")
                f.write(f"Similarity: {result['similarity']}\n")
                f.write("\nCONTENT:\n")
                f.write(f"{result['content']}\n\n")
                f.write("=" * 50 + "\n\n")

        # Also save as JSON for programmatic access
        # json_filename = f"research_export_{query_for_filename}_{file_timestamp}.json"
        # json_filepath = os.path.join(export_dir, json_filename)

        # with open(json_filepath, "w", encoding="utf-8") as f:
        #     json.dump(export_data, f, indent=2)

        return {
            "export_data": export_data,
            "txt_filepath": filepath,
        }

    async def review_synthesis(
        self,
        compiled_sections: Dict[str, str],
        original_query: str,
        research_outline: List[Dict],
        synthesis_model: str,
    ) -> Dict[str, List[Dict]]:
        """Review the compiled synthesis and suggest edits"""
        review_prompt = {
            "role": "system",
            "content": PROMPTS["section_review_system"],
        }

        # Create context with all sections
        review_context = f"# Complete Research Report on: {original_query}\n\n"
        review_context += "## Research Outline:\n"
        for topic in research_outline:
            review_context += f"- {topic['topic']}\n"
            for subtopic in topic.get("subtopics", []):
                review_context += f"  - {subtopic}\n"
        review_context += "\n"

        # Add the full content of each section
        review_context += "## Complete Report Content by Section:\n\n"
        state = self.get_state()
        memory_stats = self._ensure_memory_stats()
        section_tokens = memory_stats.get("section_tokens", {})

        for section_title, content in compiled_sections.items():
            # Get token count for this section
            tokens = section_tokens.get(section_title, 0)
            if tokens == 0:
                tokens = await self.count_tokens(content)
                section_tokens[section_title] = tokens
                memory_stats["section_tokens"] = section_tokens
                self.update_state("memory_stats", memory_stats)

            review_context += f"### {section_title} [{tokens} tokens]\n\n"
            review_context += f"{content}\n\n"

        review_context += "\nReview this research report and respond with necessary edits with specified JSON structure. Please don't include any other text in your response but the edits."

        # Create messages array
        messages = [review_prompt, {"role": "user", "content": review_context}]

        # Generate the review
        try:
            await self.emit_status(
                "info", "Reviewing and improving the synthesis...", False
            )

            # Scale temperature based on synthesis temperature valve
            review_temperature = (
                self.valves.SYNTHESIS_TEMPERATURE * 0.5
            )  # Lower temperature for more consistent review

            # Use synthesis model for reviewing
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=review_temperature,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                review_content = response["choices"][0]["message"]["content"]

                # Parse the JSON review
                try:
                    review_json_str = review_content[
                        review_content.find("{") : review_content.rfind("}") + 1
                    ]
                    review_data = json.loads(review_json_str)
                    return review_data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing review JSON: {e}")
                    # Return a minimal structure if parsing fails
                    return {"global_edits": [], "section_edits": {}}
            else:
                return {"global_edits": [], "section_edits": {}}

        except Exception as e:
            logger.error(f"Error generating synthesis review: {e}")
            return {"global_edits": [], "section_edits": {}}

    async def apply_review_edits(
        self,
        compiled_sections: Dict[str, str],
        review_data: Dict[str, Any],
        synthesis_model: str,
    ):
        """Apply the suggested edits from the review to improve the synthesis"""
        # Create deep copy of sections to modify
        edited_sections = compiled_sections.copy()

        # Track if we made any changes
        changes_made = False

        # Apply global edits
        global_edits = review_data.get("global_edits", [])
        if global_edits:
            changes_made = True
            await self.emit_status(
                "info",
                f"Applying {len(global_edits)} global edits to synthesis...",
                False,
            )

            for edit_idx, edit in enumerate(global_edits):
                find_text = edit.get("find_text", "")
                replace_text = edit.get("replace_text", "")

                if not find_text:
                    logger.warning(f"Empty find_text in edit {edit_idx+1}, skipping")
                    continue

                # Apply to each section
                for section_title, content in edited_sections.items():
                    if find_text in content:
                        edited_sections[section_title] = content.replace(
                            find_text, replace_text
                        )
                        logger.info(
                            f"Applied edit {edit_idx+1} in section '{section_title}'"
                        )

        return edited_sections, changes_made

    async def generate_titles(self, user_message, comprehensive_answer):
        """Generate a main title and subtitle for the research report"""
        titles_prompt = {
            "role": "system",
            "content": PROMPTS["titles_system"],
        }

        # Create a context with the research query and a summary of the comprehensive answer
        titles_context = f"""Original Research Query: {user_message}
	
	Research Report Content Summary:
	{comprehensive_answer}...
	
	Generate an appropriate main title and subtitle for this research report."""

        try:
            # Get the research model for title generation
            research_model = self.get_research_model()

            # Generate titles
            response = await self.generate_completion(
                research_model,
                [titles_prompt, {"role": "user", "content": titles_context}],
                temperature=0.7,  # Allow some creativity for titles
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                titles_content = response["choices"][0]["message"]["content"]

                # Extract JSON from response
                try:
                    json_str = titles_content[
                        titles_content.find("{") : titles_content.rfind("}") + 1
                    ]
                    titles_data = json.loads(json_str)

                    main_title = titles_data.get(
                        "main_title", f"Research Report: {user_message}"
                    )
                    subtitle = titles_data.get(
                        "subtitle", "A Comprehensive Analysis and Synthesis"
                    )

                    return {"main_title": main_title, "subtitle": subtitle}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing titles JSON: {e}")
                    # Fallback to simple titles
                    return {
                        "main_title": f"Research Report: {user_message[:50]}",
                        "subtitle": "A Comprehensive Analysis and Synthesis",
                    }
            else:
                # Fallback titles
                return {
                    "main_title": f"Research Report: {user_message[:50]}",
                    "subtitle": "A Comprehensive Analysis and Synthesis",
                }

        except Exception as e:
            logger.error(f"Error generating titles: {e}")
            # Fallback titles
            return {
                "main_title": f"Research Report: {user_message[:50]}",
                "subtitle": "A Comprehensive Analysis and Synthesis",
            }

    async def generate_abstract(self, user_message, comprehensive_answer, bibliography):
        """Generate an abstract for the research report"""
        abstract_prompt = {
            "role": "system",
            "content": PROMPTS["abstract_system"],
        }

        # Create a context with the full report and bibliography information
        abstract_context = f"""Research Query: {user_message}
	
	Research Report Full Content:
	{comprehensive_answer}...
	
	Generate a concise, substantive abstract focusing on substantive content and key insights rather than how the research was conducted. Please don't include any other text in your response but the abstract.
	"""

        try:
            # Get the synthesis model for abstract generation
            synthesis_model = self.get_synthesis_model()

            # Generate abstract with 5-minute timeout
            response = await asyncio.wait_for(
                self.generate_completion(
                    synthesis_model,
                    [abstract_prompt, {"role": "user", "content": abstract_context}],
                    temperature=self.valves.SYNTHESIS_TEMPERATURE,
                    user_facing=True,
                ),
                timeout=300,  # 5 minute timeout
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                abstract = response["choices"][0]["message"]["content"]
                await self.emit_message(f"*Abstract generation complete.*\n")
                return abstract
            else:
                # Fallback abstract
                await self.emit_message(f"*Abstract generation fallback used.*\n")
                return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."

        except asyncio.TimeoutError:
            logger.error("Abstract generation timed out after 5 minutes")
            # Provide a fallback abstract
            await self.emit_message(
                f"*Abstract generation timed out, using fallback.*\n"
            )
            return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."
        except Exception as e:
            logger.error(f"Error generating abstract: {e}")
            # Fallback abstract
            await self.emit_message(f"*Abstract generation error, using fallback.*\n")
            return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."
