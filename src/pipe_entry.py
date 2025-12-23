from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeEntryMixin:
    async def process_query(
        self,
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        cycle_feedback: Optional[Dict] = None,
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Process a single search query and get results with quality filtering"""
        if self.valves.DEBUG_SEARCH:
            logger.info("SEARCH DEBUG: process_query_start query=%r", query)
        await self.emit_status("info", f"Searching for: {query}", False)

        # Sanitize the query to make it safer for search engines
        sanitized_query = await self.sanitize_query(query)
        if self.valves.DEBUG_SEARCH:
            logger.info(
                "SEARCH DEBUG: process_query_sanitized query=%r sanitized=%r",
                query,
                sanitized_query,
            )

        # Get search results for the query
        search_results = await self.search_web(sanitized_query)
        if self.valves.DEBUG_SEARCH:
            empty_url_count = sum(1 for r in search_results if not r.get("url"))
            logger.info(
                "SEARCH DEBUG: search_results_received query=%r total=%d empty_url=%d",
                query,
                len(search_results),
                empty_url_count,
            )
        if not search_results:
            await self.emit_message(f"*No results found for query: {query}*\n\n")
            return []

        # Always select the most relevant results - this adds similarity scores
        search_results = await self.select_most_relevant_results(
            search_results,
            query,
            query_embedding,
            outline_embedding,
            summary_embedding,
        )

        # Process each search result until we have enough successful results
        successful_results = []
        failed_count = 0

        # Get state for access to research outline
        state = self.get_state()
        all_topics = state.get("all_topics", [])

        # Track rejected results for logging
        rejected_results = []

        for result in search_results:
            # Stop if we've reached our target of successful results
            if len(successful_results) >= self.valves.SUCCESSFUL_RESULTS_PER_QUERY:
                break

            # Stop if we've had too many consecutive failures
            if failed_count >= self.valves.MAX_FAILED_RESULTS:
                await self.emit_message(
                    f"*Skipping remaining results for query: {query} after {failed_count} failures*\n\n"
                )
                break

            try:
                # Process the result
                processed_result = await self.process_search_result(
                    result,
                    query,
                    query_embedding,
                    outline_embedding,
                    summary_embedding,
                )

                # Make sure similarity is preserved from original result
                if "similarity" in result and "similarity" not in processed_result:
                    processed_result["similarity"] = result["similarity"]

                # Check if processing was successful (has substantial content and valid URL)
                if (
                    processed_result
                    and processed_result.get("content")
                    and len(processed_result.get("content", "")) > 200
                    and processed_result.get("valid", False)
                    and processed_result.get("url", "")
                ):
                    # Add token count if not already present
                    if "tokens" not in processed_result:
                        processed_result["tokens"] = await self.count_tokens(
                            processed_result["content"]
                        )

                    # Skip results with less than 200 tokens
                    if processed_result["tokens"] < 200:
                        logger.info(
                            f"Skipping result with only {processed_result['tokens']} tokens (less than minimum 200)"
                        )
                        continue

                    # Only apply quality filter for results with low similarity
                    if (
                        self.valves.QUALITY_FILTER_ENABLED
                        and "similarity" in processed_result
                        and processed_result["similarity"]
                        < self.valves.QUALITY_SIMILARITY_THRESHOLD
                    ):
                        # Check if result is relevant using quality filter
                        is_relevant = await self.check_result_relevance(
                            processed_result,
                            query,
                            all_topics,
                        )

                        if not is_relevant:
                            # Track rejected result
                            rejected_results.append(
                                {
                                    "url": processed_result.get("url", ""),
                                    "title": processed_result.get("title", ""),
                                    "similarity": processed_result.get("similarity", 0),
                                    "processed_result": processed_result,
                                }
                            )
                            logger.warning(
                                f"Rejected irrelevant result: {processed_result.get('url', '')}"
                            )
                            continue
                    else:
                        # Skip filter for high similarity or when filtering is disabled
                        logger.info(
                            f"Skipping quality filter for result: {processed_result.get('similarity', 0):.3f}"
                        )

                    # Add to successful results
                    successful_results.append(processed_result)

                    # Get the document title for display
                    document_title = processed_result["title"]
                    if document_title == f"'{query}'" and processed_result["url"]:
                        # Try to get a better title from the URL
                        from urllib.parse import urlparse

                        parsed_url = urlparse(processed_result["url"])
                        path_parts = parsed_url.path.split("/")
                        if path_parts[-1]:
                            file_name = path_parts[-1]
                            # Clean up filename to use as title
                            if file_name.endswith(".pdf"):
                                document_title = (
                                    file_name[:-4].replace("-", " ").replace("_", " ")
                                )
                            elif "." in file_name:
                                document_title = (
                                    file_name.split(".")[0]
                                    .replace("-", " ")
                                    .replace("_", " ")
                                )
                            else:
                                document_title = file_name.replace("-", " ").replace(
                                    "_", " "
                                )
                        else:
                            # Use domain as title if no useful path
                            document_title = parsed_url.netloc

                    # Get token count for displaying
                    token_count = processed_result.get("tokens", 0)
                    if token_count == 0:
                        token_count = await self.count_tokens(
                            processed_result["content"]
                        )

                    # Display the result to the user with improved formatting
                    if processed_result["url"]:
                        # Show full URL in the result header
                        url = processed_result["url"]

                        # Check if this is a PDF (either by extension or by content type detection)
                        if (
                            url.endswith(".pdf")
                            or "application/pdf" in url
                            or self.is_pdf_content
                        ):
                            prefix = "PDF: "
                        else:
                            prefix = "Site: "

                        result_text = (
                            f"#### {prefix}{url}\n**Tokens:** {token_count}\n\n"
                        )
                    else:
                        result_text = (
                            f"#### {document_title} [{token_count} tokens]\n\n"
                        )

                    result_text += f"*Search query: {query}*\n\n"

                    # Format content with short line merging
                    content_to_display = processed_result["content"][
                        : self.valves.MAX_RESULT_TOKENS
                    ]
                    formatted_content = await self.clean_text_formatting(
                        content_to_display
                    )
                    result_text += f"{formatted_content}...\n\n"

                    # Add repeat indicator if this is a repeated URL
                    repeat_count = processed_result.get("repeat_count", 0)
                    if repeat_count > 1:
                        result_text += f"*Note: This URL has been processed {repeat_count} times*\n\n"

                    # Wrap verbose result in a collapsible block to reduce clutter in the UI
                    summary_label = (
                        (url if processed_result.get("url") else document_title)
                        or "Result"
                    )
                    if len(summary_label) > 120:
                        summary_label = summary_label[:117] + "..."

                    collapsible_result = (
                        f"<details>\n<summary>{summary_label}</summary>\n\n"
                        f"{result_text}</details>"
                    )

                    await self.emit_message(collapsible_result)

                    # Reset failed count on success
                    failed_count = 0
                else:
                    # Count as a failure
                    failed_count += 1
                    logger.warning(
                        f"Failed to get substantial content from result {len(successful_results) + failed_count} for query: {query}"
                    )

            except Exception as e:
                # Count as a failure
                failed_count += 1
                logger.error(f"Error processing result for query '{query}': {e}")
                await self.emit_message(
                    f"*Error processing a result for query: {query}*\n\n"
                )

        # If we didn't get any successful results but had rejected ones, use the top rejected result
        if not successful_results and rejected_results:
            # Sort rejected results by similarity (descending)
            sorted_rejected = sorted(
                rejected_results, key=lambda x: x.get("similarity", 0), reverse=True
            )
            top_rejected = sorted_rejected[0]

            logger.info(
                f"Using top rejected result as fallback: {top_rejected.get('url', '')}"
            )

            # Get the processed result directly from the rejection record
            if "processed_result" in top_rejected:
                processed_result = top_rejected["processed_result"]
                successful_results.append(processed_result)

                # Display the result with a note that it might not be fully relevant
                document_title = processed_result.get("title", f"Result for '{query}'")
                token_count = processed_result.get(
                    "tokens", 0
                ) or await self.count_tokens(processed_result["content"])
                url = processed_result.get("url", "")

                result_text = f"#### {document_title} [{token_count} tokens]\n\n"
                if url:
                    result_text = f"#### {'PDF: ' if url.endswith('.pdf') else 'Site: '}{url}\n**Tokens:** {token_count}\n\n"

                result_text += f"*Search query: {query}*\n\n"
                result_text += f"*Note: This result was initially filtered but is used as a fallback.*\n\n"

                # Format content
                content_to_display = processed_result["content"][
                    : self.valves.MAX_RESULT_TOKENS
                ]
                formatted_content = await self.clean_text_formatting(content_to_display)
                result_text += f"{formatted_content}...\n\n"

                summary_label = (url or document_title) or "Result"
                if len(summary_label) > 120:
                    summary_label = summary_label[:117] + "..."

                collapsible_result = (
                    f"<details>\n<summary>{summary_label}</summary>\n\n"
                    f"{result_text}</details>"
                )

                await self.emit_message(collapsible_result)

        # If we still didn't get any successful results, log this
        if not successful_results:
            logger.warning(f"No valid results obtained for query: {query}")
            await self.emit_message(f"*No valid results found for query: {query}*\n\n")

        # Update token counts with new results
        await self.update_token_counts(successful_results)

        if self.valves.DEBUG_SEARCH:
            logger.info(
                "SEARCH DEBUG: process_query_summary query=%r successful=%d failed=%d rejected=%d",
                query,
                len(successful_results),
                failed_count,
                len(rejected_results),
            )

        return successful_results

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__current_event_call__ = __event_call__
        self.__user__ = User(**__user__)
        self.__model__ = __model__
        self.__request__ = __request__

        # Extract conversation ID from the message history
        messages = body.get("messages", [])
        if not messages:
            return ""

        # Prefer OpenWebUI chat id if provided; fall back to first message id
        user_id = __user__.get("id", "anonymous")
        conversation_id = self._resolve_conversation_id(body, user_id)
        self._set_conversation_context(conversation_id)

        # Respect max concurrent conversations
        slot_acquired = await self._acquire_conversation_slot(conversation_id)
        if not slot_acquired:
            return ""

        # Ensure per-conversation executor exists
        self._ensure_executor(conversation_id)

        research_finished = False
        cleanup_done = False

        try:

            # Check if this appears to be a completely new conversation
            state = self.get_state()
            waiting_for_outline_feedback = state.get("waiting_for_outline_feedback", False)
            if (
                len(messages) <= 2 and not waiting_for_outline_feedback
            ):  # Check we're not waiting for feedback
                logger.info(f"New conversation detected with ID: {conversation_id}")
                self.reset_state(conversation_id)  # Reset all state for this conversation

            # Initialize master source table if not exists
            state = self.get_state()
            self._ensure_tracking_maps()
            self._ensure_memory_stats()

            # If the pipe is disabled or it's not a default task, return
            if not self.valves.ENABLED or (__task__ and __task__ != TASKS.DEFAULT):
                return ""

            # Get user query from the latest message
            user_message = messages[-1].get("content", "").strip()
            if not user_message:
                return ""

            # Set research date
            from datetime import datetime

            self.research_date = datetime.now().strftime("%Y-%m-%d")

            # Get state for this conversation
            state = self.get_state()

            # Check waiting flag directly in state
            if state.get("waiting_for_outline_feedback", False):
                # We're expecting outline feedback - capture the core outline data
                # to ensure it's not lost in state transitions
                feedback_data = state.get("outline_feedback_data", {})
                if feedback_data:
                    # Process the user's feedback
                    self.update_state("waiting_for_outline_feedback", False)
                    feedback_result = await self.process_outline_feedback_continuation(
                        user_message
                    )

                    # Get the research state parameters directly from feedback data
                    original_query = feedback_data.get("original_query", "")
                    outline_items = feedback_data.get("outline_items", [])
                    flat_items = feedback_data.get("flat_items", [])

                    # Retrieve all_topics and outline_embedding if we have them
                    all_topics = []
                    for topic_item in outline_items:
                        all_topics.append(topic_item["topic"])
                        all_topics.extend(topic_item.get("subtopics", []))

                    # Update outline embedding based on all_topics
                    outline_text = " ".join(all_topics)
                    outline_embedding = await self.get_embedding(outline_text)

                    # Continue the research process from the outline feedback
                    research_outline, all_topics, outline_embedding = (
                        await self.continue_research_after_feedback(
                            feedback_result,
                            original_query,
                            outline_items,
                            all_topics,
                            outline_embedding,
                        )
                    )

                    # Now continue with the main research process using the updated research state
                    user_message = original_query

                    # Initialize research state consistently
                    await self.initialize_research_state(
                        user_message,
                        research_outline,
                        all_topics,
                        outline_embedding,
                    )

                    # Update token counts
                    await self.update_token_counts()
                else:
                    # If we're supposedly waiting for feedback but have no data,
                    # treat as normal query (recover from error state)
                    self.update_state("waiting_for_outline_feedback", False)
                    logger.warning("Waiting for outline feedback but no data available")

            # Check if this is a follow-up query
            is_follow_up = await self.is_follow_up_query(messages)
            self.update_state("follow_up_mode", is_follow_up)
    
            # Get summary embedding if this is a follow-up
            summary_embedding = None
            if is_follow_up:
                prev_comprehensive_summary = state.get("prev_comprehensive_summary", "")
                if prev_comprehensive_summary:
                    try:
                        await self.emit_status(
                            "info", "Processing follow-up query...", False
                        )
                        summary_embedding = await self.get_embedding(
                            prev_comprehensive_summary
                        )
                        await self.emit_message("## Deep Research Mode: Follow-up\n\n")
                        await self.emit_message(
                            "I'll continue researching based on your follow-up query while considering our previous findings.\n\n"
                        )
                    except Exception as e:
                        logger.error(f"Error getting summary embedding: {e}")
                        # Continue without the summary embedding if there's an error
                        is_follow_up = False
                        self.update_state("follow_up_mode", False)
                        await self.emit_message("## Deep Research Mode: Activated\n\n")
                        await self.emit_message(
                            "I'll search for comprehensive information about your query. This might take a moment...\n\n"
                        )
                else:
                    is_follow_up = False
                    self.update_state("follow_up_mode", False)
            else:
                await self.emit_status("info", "Starting deep research...", False)
                await self.emit_message("## Deep Research Mode: Activated\n\n")
                await self.emit_message(
                    "I'll search for comprehensive information about your query. This might take a moment...\n\n"
                )
    
            # Check if we have research state from previous feedback
            research_state = state.get("research_state")
            if research_state:
                # Use the existing research state from feedback
                research_outline = research_state.get("research_outline", [])
                all_topics = research_state.get("all_topics", [])
                outline_embedding = research_state.get("outline_embedding")
                user_message = research_state.get("user_message", user_message)
    
                await self.emit_status(
                    "info", "Continuing research with updated outline...", False
                )
    
                # Skip to research cycles
                initial_results = []  # We'll regenerate search results
    
            else:
                # For follow-up queries, we need to generate a new research outline based on the previous summary
                if is_follow_up:
                    outline_embedding = await self.get_embedding(
                        user_message
                    )  # Create initial placeholder
                    # Step 1: Generate initial search queries for follow-up considering previous summary
                    await self.emit_status(
                        "info", "Generating initial search queries for follow-up...", False
                    )
    
                    initial_query_prompt = {
                        "role": "system",
                        "content": """You are a post-grad research assistant generating effective search queries for continued research based on an existing report.
        Based on the user's follow-up question and the previous research summary, generate 6 initial search queries.
        Each query should be specific, use relevant keywords, and be designed to find new information that builds on the previous research towards the new query.
        Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.
    
        Respond ONLY with a valid JSON object and nothing else (no prose, no code fences, no metadata, no model names, no IDs).
        JSON schema:
        {"queries": [
          "search query 1",
          "search query 2",
          "search query 3"
        ]}
        If unsure, return a minimal valid JSON with one query derived from the user message.
        Start the response with '{' and end with '}'.""",
                    }
    
                    initial_query_messages = [
                        initial_query_prompt,
                        {
                            "role": "user",
                            "content": f"Follow-up question: {user_message}\n\nPrevious research summary:\n{state.get('prev_comprehensive_summary', '')}...\n\nGenerate initial search queries for the follow-up question that build on the previous research.",
                        },
                    ]
    
                    # Get initial search queries
                    query_response = await self.generate_completion(
                        self.get_research_model(),
                        initial_query_messages,
                        temperature=self.valves.TEMPERATURE,
                        user_facing=True,
                    )
                    query_content = query_response["choices"][0]["message"]["content"]
    
                    # Extract JSON from response
                    try:
                        query_json_str = query_content[
                            query_content.find("{") : query_content.rfind("}") + 1
                        ]
                        query_data = json.loads(query_json_str)
                        raw_queries = query_data.get("queries", [])
    
                        # Normalize different possible formats to a list of strings
                        initial_queries: List[str] = []
                        if isinstance(raw_queries, list):
                            for q in raw_queries:
                                if isinstance(q, str):
                                    initial_queries.append(q)
                                elif isinstance(q, dict):
                                    candidate = q.get("query")
                                    if isinstance(candidate, str):
                                        initial_queries.append(candidate)
    
                        # If we didn't get any usable queries from JSON, trigger fallback
                        if not initial_queries or any(
                            self._looks_like_metadata_query(q) for q in initial_queries
                        ):
                            raise ValueError("No valid queries in JSON response")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing query JSON: {e}")
                        # Fallback: robust extraction from raw text
                        initial_queries = self._extract_queries_from_llm_response(
                            query_content, user_message, max_queries=3
                        )
    
                    # Display the queries to the user
                    await self.emit_message(f"### Initial Follow-up Research Queries\n\n")
                    for i, query in enumerate(initial_queries):
                        await self.emit_message(f"**Query {i+1}**: {query}\n\n")
    
                    # Execute initial searches with the follow-up queries
                    # Use summary embedding for context relevance
                    initial_results = []
                    initial_seen_urls = set()  # Track URLs seen during initial research
    
                    for query in initial_queries:
                        # Get query embedding for content comparison
                        try:
                            await self.emit_status(
                                "info", f"Getting embedding for query: {query}", False
                            )
                            query_embedding = await self.get_embedding(query)
                            if not query_embedding:
                                # If we can't get an embedding from the model, create a default one
                                logger.warning(
                                    f"Failed to get embedding for '{query}', using default"
                                )
                                fallback_dim = self.embedding_dim or 384
                                query_embedding = [0.0] * fallback_dim
                        except Exception as e:
                            logger.error(f"Error getting embedding: {e}")
                            fallback_dim = self.embedding_dim or 384
                            query_embedding = [0.0] * fallback_dim
    
                        # Process the query and get results
                        results = await self.process_query(
                            query,
                            query_embedding,
                            outline_embedding,
                            None,
                            summary_embedding,
                        )
    
                        # Filter out any URLs we've already seen in initial research
                        filtered_results = []
                        for result in results:
                            url = result.get("url", "")
                            if url and url not in initial_seen_urls:
                                filtered_results.append(result)
                                initial_seen_urls.add(url)  # Mark this URL as seen
                            else:
                                logger.info(
                                    f"Filtering out repeated URL in initial research: {url}"
                                )
    
                        # If we filtered out all results, log it
                        if results and not filtered_results:
                            logger.info(
                                f"All {len(results)} results filtered due to URL repetition in initial research"
                            )
                            # If all results were filtered, try to get at least one result by using the first one
                            if results:
                                filtered_results.append(results[0])
                                logger.info(
                                    f"Added back one result to ensure minimal research data"
                                )
    
                        # Add non-repeated results to our collection
                        initial_results.extend(filtered_results)
    
                    # Generate research outline that incorporates previous findings and new follow-up
                    await self.emit_status(
                        "info", "Generating research outline for follow-up...", False
                    )
    
                    outline_prompt = {
                        "role": "system",
                        "content": """You are a post-grad research assistant creating a structured research outline.
    	Based on the user's follow-up question, previous research summary, and new search results, create a comprehensive outline 
    	that builds on the previous research while addressing the new aspects from the follow-up question.
    	
    	The outline should:
    	1. Include relevant topics from the previous research that provide context
    	2. Add new topics that specifically address the follow-up question
    	3. Be organized in a hierarchical structure with main topics and subtopics
    	4. Focus on aspects that weren't covered in depth in the previous research
    	
    	Format your response as a valid JSON object with the following structure:
    	{"outline": [
    	  {"topic": "Main topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]},
    	  {"topic": "Main topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}
    	]}""",
                    }
    
                    # Build context from initial search results and previous summary
                    outline_context = "### Previous Research Summary:\n\n"
                    outline_context += (
                        f"{state.get('prev_comprehensive_summary', '')}...\n\n"
                    )
    
                    outline_context += "### New Search Results:\n\n"
                    for i, result in enumerate(initial_results):
                        outline_context += f"Result {i+1} (Query: '{result['query']}')\n"
                        outline_context += f"Title: {result['title']}\n"
                        outline_context += f"Content: {result['content']}...\n\n"
    
                    outline_messages = [
                        outline_prompt,
                        {
                            "role": "user",
                            "content": f"Follow-up question: {user_message}\n\n{outline_context}\n\nGenerate a comprehensive research outline that builds on previous research while addressing the follow-up question.",
                        },
                    ]
    
                    # Generate the research outline
                    outline_response = await self.generate_completion(
                        self.get_research_model(), outline_messages
                    )
                    outline_content = outline_response["choices"][0]["message"]["content"]
    
                    # Extract JSON from response
                    try:
                        outline_json_str = outline_content[
                            outline_content.find("{") : outline_content.rfind("}") + 1
                        ]
                        outline_data = json.loads(outline_json_str)
                        research_outline = outline_data.get("outline", [])
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing outline JSON: {e}")
                        # Fallback: create a simple outline if JSON parsing fails
                        research_outline = [
                            {
                                "topic": "Follow-up Information",
                                "subtopics": ["Key Aspects", "New Developments"],
                            },
                            {
                                "topic": "Extended Analysis",
                                "subtopics": ["Additional Details", "Further Examples"],
                            },
                        ]
    
                    # Create a flat list of all topics for tracking
                    all_topics = []
                    for topic_item in research_outline:
                        all_topics.append(topic_item["topic"])
                        all_topics.extend(topic_item.get("subtopics", []))
    
                    # Create outline embedding
                    outline_text = " ".join(all_topics)
                    outline_embedding = await self.get_embedding(outline_text)
    
                    # Initialize research dimensions
                    await self.initialize_research_dimensions(all_topics, user_message)
    
                    # Display the outline to the user
                    outline_text = "### Research Outline for Follow-up\n\n"
                    for topic in research_outline:
                        outline_text += f"**{topic['topic']}**\n"
                        for subtopic in topic.get("subtopics", []):
                            outline_text += f"- {subtopic}\n"
                        outline_text += "\n"
    
                    await self.emit_message(outline_text)
                    await self.emit_message(
                        "\n*Continuing with research based on this outline and previous findings...*\n\n"
                    )
    
                else:
                    # Regular new query - generate initial search queries
                    await self.emit_status(
                        "info", "Generating initial search queries...", False
                    )
    
                    initial_query_prompt = {
                        "role": "system",
                        "content": f"""You are a post-grad research assistant generating effective search queries.
        The user has submitted a research query: "{user_message}".
        Based on the user's input, generate 8 initial search queries to begin research and help us delineate the research topic.
        Half of the queries should be broad, aimed at identifying and defining the main topic and returning core characteristic information about it.
        The other half should be more specific, designed to find information to help expand on known base details of the user's query.
        Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.
    
        Respond ONLY with a valid JSON object and nothing else (no prose, no code fences, no metadata, no model names, no IDs).
        JSON schema:
        {{"queries": [
          "search query 1",
          "search query 2",
          "search query 3..."
        ]}}
        If unsure, return a minimal valid JSON with one query derived from the user message.
        Start the response with '{{' and end with '}}'.""",
                    }
    
                    initial_query_messages = [
                        initial_query_prompt,
                        {
                            "role": "user",
                            "content": f"Generate initial search queries for this user query: {user_message}",
                        },
                    ]
    
                    # Get initial search queries
                    query_response = await self.generate_completion(
                        self.get_research_model(),
                        initial_query_messages,
                        temperature=self.valves.TEMPERATURE,
                        user_facing=True,
                    )
                    query_content = query_response["choices"][0]["message"]["content"]
    
                    # Extract JSON from response
                    try:
                        query_json_str = query_content[
                            query_content.find("{") : query_content.rfind("}") + 1
                        ]
                        query_data = json.loads(query_json_str)
                        raw_queries = query_data.get("queries", [])
    
                        # Normalize different possible formats to a list of strings
                        initial_queries: List[str] = []
                        if isinstance(raw_queries, list):
                            for q in raw_queries:
                                if isinstance(q, str):
                                    initial_queries.append(q)
                                elif isinstance(q, dict):
                                    candidate = q.get("query")
                                    if isinstance(candidate, str):
                                        initial_queries.append(candidate)
    
                        # If we didn't get any usable queries from JSON, trigger fallback
                        if not initial_queries or any(
                            self._looks_like_metadata_query(q) for q in initial_queries
                        ):
                            raise ValueError("No valid queries in JSON response")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing query JSON: {e}")
                        # Fallback: robust extraction from raw text
                        initial_queries = self._extract_queries_from_llm_response(
                            query_content, user_message, max_queries=3
                        )
    
                    # Display the queries to the user
                    await self.emit_message(f"### Initial Research Queries\n\n")
                    for i, query in enumerate(initial_queries):
                        await self.emit_message(f"**Query {i+1}**: {query}\n\n")
    
                    # Step 2: Execute initial searches and collect results
                    # Get outline embedding (placeholder - will be updated after outline is created)
                    outline_embedding = await self.get_embedding(user_message)
    
                    initial_results = []
                    for query in initial_queries:
                        # Get query embedding for content comparison
                        try:
                            await self.emit_status(
                                "info", f"Getting embedding for query: {query}", False
                            )
                            query_embedding = await self.get_embedding(query)
                            if not query_embedding:
                                # If we can't get an embedding from the model, create a default one
                                logger.warning(
                                    f"Failed to get embedding for '{query}', using default"
                                )
                                fallback_dim = self.embedding_dim or 384
                                query_embedding = [0.0] * fallback_dim
                        except Exception as e:
                            logger.error(f"Error getting embedding: {e}")
                            fallback_dim = self.embedding_dim or 384
                            query_embedding = [0.0] * fallback_dim
    
                        # Process the query and get results
                        results = await self.process_query(
                            query,
                            query_embedding,
                            outline_embedding,
                            None,
                            summary_embedding,
                        )
    
                        # Add successful results to our collection
                        initial_results.extend(results)
    
                    # Check if we got any useful results
                    useful_results = [
                        r for r in initial_results if len(r.get("content", "")) > 200
                    ]
    
                    # If we didn't get any useful results, create a minimal result to continue
                    if not useful_results:
                        await self.emit_message(
                            f"*Unable to find initial search results. Creating research outline based on the query alone.*\n\n"
                        )
                        initial_results = [
                            {
                                "title": f"Information about {user_message}",
                                "url": "",
                                "content": f"This is a placeholder for research about {user_message}. The search failed to return usable results.",
                                "query": user_message,
                            }
                        ]
                    else:
                        # Log the successful results
                        logger.info(
                            f"Found {len(useful_results)} useful results from initial queries"
                        )
    
                    # Step 3: Generate research outline based on user query AND initial results
                    await self.emit_status(
                        "info",
                        "Analyzing initial results and generating research outline...",
                        False,
                    )
    
                    outline_prompt = {
                        "role": "system",
                        "content": f"""You are a post-graduate academic scholar tasked with creating a structured research outline.
    	Based on the user's query and the initial search results, create a comprehensive conceptual outline of additional information 
    	needed to completely and thoroughly address the user's original query: "{user_message}".
    	
    	The outline must:
    	1. Break down the query into key concepts that need to be researched and key details about important figures, details, methods, etc.
    	2. Be organized in a hierarchical structure, with main topics directly relevant to addressing the query, and subtopics to flesh out main topics.
    	3. Include topics discovered in the initial search results relevant to addressing the user's input, while ignoring overly-specific or unrelated topics.
    
        The outline MUST NOT:
        1. Delve into philosophical or theoretical approaches, unless clearly appropriate to the subject or explicitly solicited by the user.
        2. Include generic topics or subtopics, i.e. "considering complexities" or "understanding the question".
        3. Reflect your own opinions, bias, notions, priorities, or other non-academic impressions of the area of research.
    
        Your outline should conceptually take up the entire space between an introduction and conclusion, filling in the entirety of the research volume.
        Do NOT allow rendering artifacts, web site UI features, HTML/CSS/underlying website build language, or any other irrelevant text to distract you from your goal.
        Don't add an appendix topic, nor an explicit introduction or conclusion topic. ONLY include the outline in your response.
    	
    	Format your response as a valid JSON object with the following structure:
    	{{"outline": [
    	  {{"topic": "Main topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]}},
    	  {{"topic": "Main topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}}
    	]}}""",
                    }
    
                    # Build context from initial search results
                    outline_context = "### Initial Search Results:\n\n"
                    for i, result in enumerate(initial_results):
                        outline_context += f"Result {i+1} (Query: '{result['query']}')\n"
                        outline_context += f"Title: {result['title']}\n"
                        outline_context += f"Content: {result['content']}...\n\n"
    
                    outline_messages = [
                        outline_prompt,
                        {
                            "role": "user",
                            "content": f"Original query: {user_message}\n\n{outline_context}\n\nGenerate a structured research outline following the instructions in the system prompt. ",
                        },
                    ]
    
                    # Generate the research outline
                    outline_response = await self.generate_completion(
                        self.get_research_model(), outline_messages
                    )
                    outline_content = outline_response["choices"][0]["message"]["content"]
    
                    # Extract JSON from response
                    try:
                        outline_json_str = outline_content[
                            outline_content.find("{") : outline_content.rfind("}") + 1
                        ]
                        outline_data = json.loads(outline_json_str)
                        research_outline = outline_data.get("outline", [])
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing outline JSON: {e}")
                        # Fallback: create a simple outline if JSON parsing fails
                        research_outline = [
                            {
                                "topic": "General Information",
                                "subtopics": ["Background", "Key Concepts"],
                            },
                            {
                                "topic": "Specific Aspects",
                                "subtopics": ["Detailed Analysis", "Examples"],
                            },
                        ]
    
                    # Create a flat list of all topics and subtopics for tracking completeness
                    all_topics = []
                    for topic_item in research_outline:
                        all_topics.append(topic_item["topic"])
                        all_topics.extend(topic_item.get("subtopics", []))
    
                    # Update outline embedding now that we have the actual outline
                    outline_text = " ".join(all_topics)
                    outline_embedding = await self.get_embedding(outline_text)
    
                    # Initialize dimension-aware research tracking
                    await self.initialize_research_dimensions(all_topics, user_message)
    
                    # User interaction for outline feedback (if enabled)
                    if self.valves.INTERACTIVE_RESEARCH:
                        # Get user feedback on the research outline
                        if not state.get("waiting_for_outline_feedback", False):
                            # Display the outline to the user
                            outline_text = "### Research Outline\n\n"
                            for topic in research_outline:
                                outline_text += f"**{topic['topic']}**\n"
                                for subtopic in topic.get("subtopics", []):
                                    outline_text += f"- {subtopic}\n"
                                outline_text += "\n"
    
                            await self.emit_message(outline_text)
    
                            # Get user feedback (this will set the flags and state for continuation)
                            feedback_result = await self.process_user_outline_feedback(
                                research_outline, user_message
                            )
    
                            # Return empty string to pause execution until next message
                            return ""
                    else:
                        # Regular display of outline if interactive research is disabled
                        # Display the outline to the user
                        outline_text = "### Research Outline\n\n"
                        for topic in research_outline:
                            outline_text += f"**{topic['topic']}**\n"
                            for subtopic in topic.get("subtopics", []):
                                outline_text += f"- {subtopic}\n"
                            outline_text += "\n"
    
                        await self.emit_message(outline_text)
    
                        # Initialize research state consistently
                        await self.initialize_research_state(
                            user_message,
                            research_outline,
                            all_topics,
                            outline_embedding,
                            initial_results,
                        )
    
                        # Update token counts
                        await self.update_token_counts(initial_results)
    
                        # Display message about continuing
                        await self.emit_message(
                            "\n*Continuing with research outline...*\n\n"
                        )
    
            # Update status to show we've moved beyond outline generation
            await self.emit_status(
                "info", "Research outline generated. Beginning research cycles...", False
            )
    
            # Initialize research variables for continued cycles
            cycle = 1  # We've already done one cycle with the initial queries
            max_cycles = self.valves.MAX_CYCLES
            min_cycles = self.valves.MIN_CYCLES
            completed_topics = set(state.get("completed_topics", set()))
            irrelevant_topics = set(state.get("irrelevant_topics", set()))
            search_history = state.get("search_history", [])
            results_history = state.get("results_history", []) + (initial_results or [])
            active_outline = list(set(all_topics) - completed_topics - irrelevant_topics)
            cycle_summaries = state.get("cycle_summaries", [])
    
            # Ensure consistent token counts
            await self.update_token_counts()
    
            # Step 4: Begin research cycles
            while cycle < max_cycles and active_outline:
                cycle += 1
                await self.emit_status(
                    "info",
                    f"Research cycle {cycle}/{max_cycles}: Generating search queries...",
                    False,
                )
    
                # Calculate research trajectory from previous cycles
                if cycle > 2 and results_history:
                    research_trajectory = await self.calculate_research_trajectory(
                        search_history, results_history
                    )
    
                    # Update research trajectory
                    self.update_state("research_trajectory", research_trajectory)
    
                # Calculate gap vector for directing research toward uncovered areas
                gap_vector = await self.calculate_gap_vector()
    
                # Rank active topics by priority using semantic analysis
                prioritized_topics = await self.rank_topics_by_research_priority(
                    active_outline, gap_vector, completed_topics, results_history
                )
    
                # Get most important topics for this cycle (limited to 10)
                priority_topics = prioritized_topics[:10]
    
                # Build context for query generation with all the improved elements
                search_context = ""
    
                # Include original query and user feedback
                search_context += f"### Original Query:\n{user_message}\n\n"
    
                # If there was user feedback, include it as clarification
                user_preferences = state.get("user_preferences", {})
                if user_preferences.get("pdv") is not None:
                    # Try to translate PDV to words
                    pdv_words = await self.translate_pdv_to_words(
                        user_preferences.get("pdv")
                    )
                    if pdv_words:
                        search_context += f"### User Preferences:\nThe user is more interested in topics related to: {pdv_words}\n\n"
    
                # Include prioritized research topics
                search_context += "### Priority research topics for this cycle:\n"
                for topic in priority_topics:
                    search_context += f"- {topic}\n"
    
                # Add a separate section for all remaining topics
                if len(active_outline) > len(priority_topics):
                    search_context += "\n### Additional topics still needing research:\n"
                    for topic in active_outline:
                        if topic not in priority_topics:
                            search_context += f"- {topic}\n"
    
                # Include recent search history (only last 3 cycles)
                if search_history:
                    search_context += "\n### Recent search queries:\n"
                    search_context += ", ".join([f"'{q}'" for q in search_history[-9:]])
                    search_context += "\n\n"
    
                # Include previous results summary
                if results_history:
                    search_context += "### Recent research results summary:\n\n"
                    # Use most recent results only
                    recent_results = results_history[-6:]  # Show just the latest 6 results
    
                    for i, result in enumerate(recent_results):
                        search_context += f"Result {i+1} (Query: '{result['query']}')\n"
                        search_context += f"URL: {result.get('url', 'No URL')}\n"
                        search_context += f"Summary: {result['content'][:2000]}...\n\n"
    
                # Include previous cycle summaries (last 3 only)
                if cycle_summaries:
                    search_context += "\n### Previous cycle summaries:\n"
                    for i, summary in enumerate(cycle_summaries[-3:]):
                        search_context += f"Cycle {cycle-3+i} Summary: {summary}\n\n"
    
                # Include identified research gaps from dimensional analysis
                research_dimensions = state.get("research_dimensions")
                if research_dimensions:
                    gaps = await self.identify_research_gaps()
                    if gaps:
                        search_context += "\n### Identified research gaps:\n"
                        for gap in gaps:
                            search_context += f"- Dimension {gap+1}\n"
    
                # Include previous comprehensive summary if this is a follow-up
                if is_follow_up and state.get("prev_comprehensive_summary"):
                    search_context += "### Previous Research Summary:\n\n"
                    summary_excerpt = state.get("prev_comprehensive_summary", "")[:5000]
                    search_context += f"{summary_excerpt}...\n\n"
    
                # Generate new queries for this cycle
                query_objects = await self.improved_query_generation(
                    user_message, priority_topics, search_context
                )
    
                # Extract query strings and topics
                current_cycle_queries = query_objects
    
                # Track topics used for queries to apply dampening in future cycles
                used_topics = [
                    query_obj.get("topic", "")
                    for query_obj in current_cycle_queries
                    if query_obj.get("topic")
                ]
                await self.update_topic_usage_counts(used_topics)
    
                # Display the queries to the user
                await self.emit_message(f"### Research Cycle {cycle}: Search Queries\n\n")
                for i, query_obj in enumerate(current_cycle_queries):
                    query = query_obj.get("query", "")
                    topic = query_obj.get("topic", "")
                    await self.emit_message(
                        f"**Query {i+1}**: {query}\n**Topic**: {topic}\n\n"
                    )
    
                # Extract query strings for search history
                query_strings = [q.get("query", "") for q in current_cycle_queries]
    
                # Add queries to search history
                search_history.extend(query_strings)
                self.update_state("search_history", search_history)
    
                # Execute searches and process results SEQUENTIALLY
                cycle_results = []
                for query_obj in current_cycle_queries:
                    query = query_obj.get("query", "")
                    topic = query_obj.get("topic", "")
    
                    # Get query embedding for content comparison
                    try:
                        query_embedding = await self.get_embedding(query)
                        if not query_embedding:
                            fallback_dim = self.embedding_dim or 384
                            query_embedding = [0.0] * fallback_dim
                    except Exception as e:
                        logger.error(f"Error getting embedding: {e}")
                        fallback_dim = self.embedding_dim or 384
                        query_embedding = [0.0] * fallback_dim
    
                    # Apply semantic transformation if available
                    semantic_transformations = state.get("semantic_transformations")
                    if semantic_transformations:
                        transformed_query = await self.apply_semantic_transformation(
                            query_embedding, semantic_transformations
                        )
                        # Use transformed embedding if available
                        if transformed_query:
                            query_embedding = transformed_query
    
                    # Process the query and get results
                    results = await self.process_query(
                        query,
                        query_embedding,
                        outline_embedding,
                        None,
                        summary_embedding,
                    )
    
                    # Add successful results to the cycle results and history
                    cycle_results.extend(results)
                    results_history.extend(results)
    
                # Update in state
                self.update_state("results_history", results_history)
    
                # Step 6: Analyze results and update research outline
                if cycle_results:
                    await self.emit_status(
                        "info",
                        "Analyzing search results and updating research outline...",
                        False,
                    )
                    analysis_prompt = {
                        "role": "system",
                        "content": f"""You are a post-grad researcher analyzing search results and updating a research outline.
    Examine the search results and the current research outline to assess the state of research.
    This is cycle {cycle} out of a maximum of {max_cycles} research cycles.
    Determine which topics have been adequately addressed by the search results.
    Update the research outline by classifying topics into different categories.
    
    Topics should be classified as:
    - COMPLETED: Topics that have been fully or reasonably addressed with researched information.
    - PARTIAL: Topics that have minimal information and need more research. Don't let topics languish here!
      If one hasn't been addressed in a while, reconsider if it actually has been, or if it's possibly irrelevant.
    - IRRELEVANT: Topics that are not actually relevant to the main query, are red herrings, 
      based on misidentified subjects, or are website artifacts rather than substantive topics.
      For example, mark as irrelevant any topics about unrelated subjects that were mistakenly
      included due to ambiguous terms, incorrect definitions for acronyms with multiple meanings,
      or page elements/advertisements from websites that don't relate to the actual query.
    - NEW: New topics discovered in the search results that should be added to the research.
      Topics that feel like a logical extension of the user's line of questioning and direction of research, 
      or that are clearly important to a specific subject but aren't currently included, belong here.
    
    Remember that the research ultimately aims to address the original query: "{user_message}".
    
    Format your response as a valid JSON object with the following structure:
    {{
      "completed_topics": ["Topic 1", "Subtopic 2.1"],
      "partial_topics": ["Topic 2"],
      "irrelevant_topics": ["Topic that's a distraction", "Misidentified subject"],
      "new_topics": ["New topic discovered"],
      "analysis": "Brief analysis of what we've learned so far with a focus on results from this past cycle"
    }}""",
                    }
    
                    # Create a context with the current outline and search results
                    analysis_context = "### Current Research Outline Topics:\n"
                    analysis_context += "\n".join(
                        [f"- {topic}" for topic in active_outline]
                    )
                    analysis_context += "\n\n### Latest Search Results:\n\n"
    
                    for i, result in enumerate(cycle_results):
                        analysis_context += f"Result {i+1} (Query: '{result['query']}')\n"
                        analysis_context += f"Title: {result['title']}\n"
                        analysis_context += f"Content: {result['content'][:2000]}...\n\n"
    
                    # Include previous cycle summaries for continuity
                    if cycle_summaries:
                        analysis_context += "\n### Previous cycle summaries:\n"
                        for i, summary in enumerate(cycle_summaries):
                            analysis_context += f"Cycle {i+1} Summary: {summary}\n\n"
    
                    # Include lists of completed and irrelevant topics
                    if completed_topics:
                        analysis_context += "\n### Already completed topics:\n"
                        for topic in completed_topics:
                            analysis_context += f"- {topic}\n"
    
                    if irrelevant_topics:
                        analysis_context += "\n### Already identified irrelevant topics:\n"
                        for topic in irrelevant_topics:
                            analysis_context += f"- {topic}\n"
    
                    # Include user preferences if applicable
                    if (
                        self.valves.USER_PREFERENCE_THROUGHOUT
                        and state.get("user_preferences", {}).get("pdv") is not None
                    ):
                        analysis_context += (
                            "\n### User preferences are being applied to research\n"
                        )
    
                    analysis_messages = [
                        analysis_prompt,
                        {
                            "role": "user",
                            "content": f"Original query: {user_message}\n\n{analysis_context}\n\nAnalyze these results and update the research outline.",
                        },
                    ]
    
                    try:
                        analysis_response = await self.generate_completion(
                            self.get_research_model(), analysis_messages
                        )
                        analysis_content = analysis_response["choices"][0]["message"][
                            "content"
                        ]
    
                        # Extract JSON from response
                        analysis_json_str = analysis_content[
                            analysis_content.find("{") : analysis_content.rfind("}") + 1
                        ]
                        analysis_data = json.loads(analysis_json_str)
    
                        # Update completed topics
                        newly_completed = set(analysis_data.get("completed_topics", []))
                        completed_topics.update(newly_completed)
                        self.update_state("completed_topics", completed_topics)
    
                        # Update irrelevant topics
                        newly_irrelevant = set(analysis_data.get("irrelevant_topics", []))
                        irrelevant_topics.update(newly_irrelevant)
                        self.update_state("irrelevant_topics", irrelevant_topics)
    
                        # Add any new topics discovered
                        new_topics = analysis_data.get("new_topics", [])
                        for topic in new_topics:
                            if (
                                topic not in all_topics
                                and topic not in completed_topics
                                and topic not in irrelevant_topics
                            ):
                                active_outline.append(topic)
                                all_topics.append(topic)
    
                        # Update active outline by removing completed and irrelevant topics
                        active_outline = [
                            topic
                            for topic in active_outline
                            if topic not in completed_topics
                            and topic not in irrelevant_topics
                        ]
    
                        # Update in state
                        self.update_state("active_outline", active_outline)
                        self.update_state("all_topics", all_topics)
    
                        # Save the analysis summary
                        cycle_summaries.append(
                            analysis_data.get("analysis", f"Analysis for cycle {cycle}")
                        )
                        self.update_state("cycle_summaries", cycle_summaries)
    
                        # Create the current checklist for display to the user
                        current_checklist = {
                            "completed": newly_completed,
                            "partial": set(analysis_data.get("partial_topics", [])),
                            "irrelevant": newly_irrelevant,
                            "new": set(new_topics),
                            "remaining": set(active_outline),
                        }
    
                        # Display analysis to the user, localized based on OUTPUT_LANGUAGE
                        output_lang = (self.valves.OUTPUT_LANGUAGE or "auto").lower()
                        is_pl = output_lang in ("pl", "polish")
    
                        if is_pl:
                            analysis_heading = f"### Analiza badań (cykl {cycle})\n\n"
                            base_analysis = analysis_data.get(
                                "analysis", "Analiza niedostępna."
                            )
                            completed_label = "**Zrealizowane tematy:**\n"
                            partial_label = "**Tematy częściowo opracowane:**\n"
                            irrelevant_label = (
                                "**Tematy nieistotne / rozpraszające:**\n"
                            )
                            new_label = "**Nowe odkryte tematy:**\n"
                            remaining_label = "**Pozostałe tematy:**\n"
                            more_suffix = " więcej"
                        else:
                            analysis_heading = f"### Research Analysis (Cycle {cycle})\n\n"
                            base_analysis = analysis_data.get(
                                "analysis", "Analysis not available."
                            )
                            completed_label = "**Topics Completed:**\n"
                            partial_label = "**Topics Partially Addressed:**\n"
                            irrelevant_label = "**Irrelevant/Distraction Topics:**\n"
                            new_label = "**New Topics Discovered:**\n"
                            remaining_label = "**Remaining Topics:**\n"
                            more_suffix = " more"
    
                        analysis_text = analysis_heading
                        analysis_text += f"{base_analysis}\n\n"
    
                        # Collapsible section for search results from this cycle
                        results_details = ""
                        if cycle_results:
                            for i, result in enumerate(cycle_results):
                                title = result.get("title", "N/A")
                                url = result.get("url", "N/A")
                                query = result.get("query", "N/A")
                                content_summary = result.get("content", "")
                                summary_preview = (
                                    (content_summary[:300] + "...")
                                    if len(content_summary) > 300
                                    else content_summary
                                )
    
                                results_details += f"**Result {i+1}: {title}**\n"
                                results_details += f"URL: {url}\n"
                                results_details += f"Query: '{query}'\n"
                                results_details += f"Summary: {summary_preview}\n\n"
    
                            if results_details:
                                results_summary = (
                                    f"Pokaż {len(cycle_results)} wyników z tego cyklu"
                                    if is_pl
                                    else f"View {len(cycle_results)} search results from this cycle"
                                )
                                analysis_text += (
                                    f"<details>\n<summary>{results_summary}</summary>\n\n"
                                    f"{results_details}</details>\n\n"
                                )
    
                        # Collapsible section for detailed topic updates
                        topic_details = ""
                        if newly_completed:
                            topic_details += completed_label
                            for topic in newly_completed:
                                topic_details += f"✓ {topic}\n"
                            topic_details += "\n"
    
                        if analysis_data.get("partial_topics"):
                            partial_topics = analysis_data.get("partial_topics")
                            topic_details += partial_label
                            # Show only first 5 partial topics
                            for topic in partial_topics[:5]:
                                topic_details += f"⚪ {topic}\n"
                            # Add count of additional topics if there are more than 5
                            if len(partial_topics) > 5:
                                topic_details += (
                                    f"...and {len(partial_topics) - 5}{more_suffix}\n"
                                )
                            topic_details += "\n"
    
                        # Add display for irrelevant topics
                        if newly_irrelevant:
                            topic_details += irrelevant_label
                            for topic in newly_irrelevant:
                                topic_details += f"✗ {topic}\n"
                            topic_details += "\n"
    
                        if new_topics:
                            topic_details += new_label
                            for topic in new_topics:
                                topic_details += f"+ {topic}\n"
                            topic_details += "\n"
    
                        if active_outline:
                            topic_details += remaining_label
                            for topic in active_outline[:5]:  # Show just the first 5
                                topic_details += f"□ {topic}\n"
                            if len(active_outline) > 5:
                                topic_details += (
                                    f"...and {len(active_outline) - 5}{more_suffix}\n"
                                )
                            topic_details += "\n"
    
                        if topic_details:
                            topic_summary = (
                                "Pokaż szczegółowe aktualizacje tematów"
                                if is_pl
                                else "View detailed topic updates"
                            )
                            analysis_text += (
                                f"<details>\n<summary>{topic_summary}</summary>\n\n"
                                f"{topic_details}</details>\n\n"
                            )
    
                        # Store dimension coverage in state but don't display it during cycles
                        research_dimensions = state.get("research_dimensions")
                        if research_dimensions:
                            try:
                                # Store the coverage for later display at summary
                                state["latest_dimension_coverage"] = research_dimensions[
                                    "coverage"
                                ].copy()
                                self.update_state(
                                    "latest_dimension_coverage",
                                    research_dimensions["coverage"],
                                )
                            except Exception as e:
                                logger.error(f"Error storing dimension coverage: {e}")
    
                        await self.emit_message(analysis_text)
    
                        # Update dimension coverage for each result to improve tracking
                        for result in cycle_results:
                            content = result.get("content", "")
                            if content:
                                # Use similarity to query as quality factor (0.5-1.0 range)
                                quality = 0.5
                                if "similarity" in result:
                                    quality = 0.5 + result["similarity"] * 0.5
                                await self.update_dimension_coverage(content, quality)
    
                    except Exception as e:
                        logger.error(f"Error analyzing results: {e}")
                        await self.emit_message(
                            f"### Research Progress (Cycle {cycle})\n\nContinuing research on remaining topics...\n\n"
                        )
                        # Mark one topic as completed to ensure progress
                        if active_outline:
                            # Find the most covered topic based on similarities to gathered results
                            topic_scores = {}
    
                            # Only attempt similarity analysis if we have results
                            if cycle_results:
                                for topic in active_outline:
                                    topic_embedding = await self.get_embedding(topic)
                                    if topic_embedding:
                                        # Calculate similarity to each result
                                        topic_score = 0.0
                                        for result in cycle_results:
                                            content = result.get("content", "")[
                                                :1000
                                            ]  # Use first 1000 chars
                                            content_embedding = await self.get_embedding(
                                                content
                                            )
                                            if content_embedding:
                                                sim = cosine_similarity(
                                                    [topic_embedding], [content_embedding]
                                                )[0][0]
                                                topic_score += sim
    
                                        # Average the score
                                        if cycle_results:
                                            topic_score /= len(cycle_results)
    
                                        topic_scores[topic] = topic_score
    
                            # If we have scores, select the highest; otherwise just take the first one
                            if topic_scores:
                                completed_topic = max(
                                    topic_scores.items(), key=lambda x: x[1]
                                )[0]
                                logger.info(
                                    f"Selected most covered topic: {completed_topic} (score: {topic_scores[completed_topic]:.3f})"
                                )
                            else:
                                completed_topic = active_outline[0]
                                logger.info(
                                    f"No similarity data available, selected first topic: {completed_topic}"
                                )
    
                            completed_topics.add(completed_topic)
                            self.update_state("completed_topics", completed_topics)
    
                            active_outline.remove(completed_topic)
                            self.update_state("active_outline", active_outline)
    
                            await self.emit_message(
                                f"**Topic Addressed:** {completed_topic}\n\n"
                            )
                            # Add minimal analysis to cycle summaries
                            cycle_summaries.append(f"Completed topic: {completed_topic}")
                            self.update_state("cycle_summaries", cycle_summaries)
    
                # Check termination criteria
                if not active_outline or active_outline == []:
                    await self.emit_status(
                        "info", "All research topics have been addressed!", False
                    )
                    break
    
                if cycle >= min_cycles and len(completed_topics) / len(all_topics) > 0.7:
                    await self.emit_status(
                        "info",
                        "Most research topics have been addressed. Finalizing...",
                        False,
                    )
                    break
    
                # Continue to next cycle if we haven't hit max_cycles
                if cycle >= max_cycles:
                    await self.emit_status(
                        "info",
                        f"Maximum research cycles ({max_cycles}) reached. Finalizing...",
                        False,
                    )
                    break
    
                await self.emit_status(
                    "info",
                    f"Research cycle {cycle} complete. Moving to next cycle...",
                    False,
                )
    
            # Apply stepped compression to all research results if enabled
            if self.valves.STEPPED_SYNTHESIS_COMPRESSION and len(results_history) > 2:
                await self.emit_status(
                    "info", "Applying stepped compression to research results...", False
                )
    
                # Track token counts before compression
                total_tokens_before = 0
                for result in results_history:
                    tokens = await self.count_tokens(result.get("content", ""))
                    total_tokens_before += tokens
    
                # Apply stepped compression to results
                results_history = await self.apply_stepped_compression(
                    results_history,
                    query_embedding if "query_embedding" in locals() else None,
                    summary_embedding,
                )
    
                # Calculate total tokens after compression
                total_tokens_after = sum(
                    result.get("tokens", 0) for result in results_history
                )
    
                # Log token reduction
                token_reduction = total_tokens_before - total_tokens_after
                if total_tokens_before > 0:
                    percent_reduction = (token_reduction / total_tokens_before) * 100
                    logger.info(
                        f"Stepped compression: {total_tokens_before} → {total_tokens_after} tokens "
                        f"(saved {token_reduction} tokens, {percent_reduction:.1f}% reduction)"
                    )
    
                    await self.emit_status(
                        "info",
                        f"Compressed research results from {total_tokens_before} to {total_tokens_after} tokens",
                        False,
                    )
    
            # Step 7: Generate refined synthesis outline
            await self.emit_status(
                "info", "Generating refined outline for synthesis...", False
            )
    
            synthesis_outline = await self.generate_synthesis_outline(
                research_outline, completed_topics, user_message, results_history
            )
    
            # If synthesis outline generation failed, use original
            if not synthesis_outline:
                synthesis_outline = research_outline
    
            # Step 8: Synthesize final answer with the selected model - Section by Section with citations
            await self.emit_synthesis_status(
                "Synthesizing comprehensive answer from research results..."
            )
            await self.emit_message("\n\n---\n\n### Research Complete\n\n")
    
            # Make sure dimensions data is up-to-date
            await self.update_research_dimensions_display()
    
            # Display the final research outline first
            await self.emit_message("### Final Research Outline\n\n")
            for topic_item in synthesis_outline:
                topic = topic_item["topic"]
                subtopics = topic_item.get("subtopics", [])
    
                await self.emit_message(f"**{topic}**\n")
                for subtopic in subtopics:
                    await self.emit_message(f"- {subtopic}\n")
                await self.emit_message("\n")
    
            # Display research dimensions after the outline
            await self.emit_status(
                "info", "Displaying research dimensions coverage...", False
            )
            await self.emit_message("### Research Dimensions (Ordered)\n\n")
    
            research_dimensions = state.get("research_dimensions")
            latest_coverage = state.get("latest_dimension_coverage")
    
            if latest_coverage and research_dimensions:
                try:
                    # Translate dimensions to words
                    dimension_labels = await self.translate_dimensions_to_words(
                        research_dimensions, latest_coverage
                    )
    
                    # Sort dimensions by coverage (highest to lowest)
                    sorted_dimensions = sorted(
                        dimension_labels, key=lambda x: x.get("coverage", 0), reverse=True
                    )
    
                    # Display dimensions without coverage percentages
                    for dim in sorted_dimensions[:10]:  # Limit to top 10
                        await self.emit_message(f"- {dim.get('words', 'Dimension')}\n")
    
                    await self.emit_message("\n")
                except Exception as e:
                    logger.error(f"Error displaying final dimension coverage: {e}")
                    await self.emit_message("*Error displaying research dimensions*\n\n")
            else:
                logger.warning("No research dimensions data available for display")
                await self.emit_message("*No research dimension data available*\n\n")
    
            # Determine which model to use for synthesis
            synthesis_model = self.get_synthesis_model()
            await self.emit_synthesis_status(
                f"Using {synthesis_model} for section generation..."
            )
    
            # Clear section content storage
            self.update_state("section_synthesized_content", {})
            self.update_state("subtopic_synthesized_content", {})
            self.update_state("section_sources_map", {})
            self.update_state("section_citations", {})
    
            # Initialize global citation map if not exists
            if "global_citation_map" not in state:
                self.update_state("global_citation_map", {})
    
            # Process each main topic and its subtopics
            compiled_sections = {}
    
            # Include only main topics that are not in irrelevant_topics
            relevant_topics = [
                topic
                for topic in synthesis_outline
                if topic["topic"] not in irrelevant_topics
            ]
    
            # If we have no relevant topics, use a simple structure
            if not relevant_topics:
                relevant_topics = [
                    {"topic": "Research Summary", "subtopics": ["General Information"]}
                ]
    
            # Initialize _seen_sections and _seen_subtopics attributes
            self._seen_sections = set()
            self._seen_subtopics = set()
    
            # Generate content for each section with proper status updates
            all_verified_citations = []
            all_flagged_citations = []
    
            for topic_item in relevant_topics:
                section_title = topic_item["topic"]
                subtopics = [
                    st
                    for st in topic_item.get("subtopics", [])
                    if st not in irrelevant_topics
                ]
    
                # Generate content for this section with inline citations (subtopic-based)
                section_data = await self.generate_section_content_with_citations(
                    section_title,
                    subtopics,
                    user_message,
                    results_history,
                    synthesis_model,
                    is_follow_up,
                    state.get("prev_comprehensive_summary", "") if is_follow_up else "",
                )
    
                # Store in compiled sections
                compiled_sections[section_title] = section_data["content"]
    
                # Track citations for bibliography generation
                if "verified_citations" in section_data:
                    all_verified_citations.extend(
                        section_data.get("verified_citations", [])
                    )
                if "flagged_citations" in section_data:
                    all_flagged_citations.extend(section_data.get("flagged_citations", []))
    
            # Store verification results for later use
            verification_results = {
                "verified": all_verified_citations,
                "flagged": all_flagged_citations,
            }
            self.update_state("verification_results", verification_results)
    
            # Process any non-standard citations that might still be in the text
            await self.emit_synthesis_status("Processing additional citation formats...")
            additional_citations = []
            master_source_table = state.get("master_source_table", {})
            global_citation_map = state.get("global_citation_map", {})
    
            for section_title, content in compiled_sections.items():
                # Use existing method to find non-standard citations
                section_citations = await self.identify_and_correlate_citations(
                    section_title, content, master_source_table
                )
    
                if section_citations:
                    # Add these citations to our tracking
                    additional_citations.extend(section_citations)
    
                    # Add to section citations tracking
                    all_section_citations = state.get("section_citations", {})
                    if section_title not in all_section_citations:
                        all_section_citations[section_title] = []
                    all_section_citations[section_title].extend(section_citations)
                    self.update_state("section_citations", all_section_citations)
    
                    # Add URLs to global citation map
                    for citation in section_citations:
                        url = citation.get("url", "")
                        if url and url not in global_citation_map:
                            global_citation_map[url] = len(global_citation_map) + 1
    
            # Update global citation map with any new URLs found
            self.update_state("global_citation_map", global_citation_map)
    
            # Final pass to handle non-standard citations and apply strikethrough
            await self.emit_synthesis_status("Finalizing citation formatting...")
            for section_title, content in list(compiled_sections.items()):
                modified_content = content

                # Handle only non-standard citations (numeric ones were already processed)
                section_citations = [
                    c for c in additional_citations if c.get("section") == section_title
                ]

                for citation in section_citations:
                    url = citation.get("url", "")
                    raw_text = citation.get("raw_text", "")

                    if url and url in global_citation_map and raw_text:
                        global_id = global_citation_map[url]
                        # Replace the original citation text with the global ID
                        modified_content = modified_content.replace(
                            raw_text, f"[{global_id}]"
                        )

                # Update the original section content
                compiled_sections[section_title] = modified_content

            # Sync updated compiled_sections back into section_synthesized_content
            # so downstream consumers (e.g., bibliography, verification) see
            # the final citation-formatted version.
            state = self.get_state()
            section_synthesized_content = state.get("section_synthesized_content", {})
            for section_title, content in compiled_sections.items():
                section_synthesized_content[section_title] = content
            self.update_state("section_synthesized_content", section_synthesized_content)

            # Generate bibliography from citation data using the finalized content
            await self.emit_synthesis_status("Generating bibliography...")
            bibliography_data = await self.generate_bibliography(
                master_source_table, global_citation_map, compiled_sections
            )
    
            # Generate titles for the report
            await self.emit_synthesis_status("Generating report titles...")
            titles = await self.generate_titles(
                user_message, "".join(compiled_sections.values())
            )
    
            # After all sections are generated, perform synthesis review
            await self.emit_synthesis_status("Reviewing and improving the synthesis...")
    
            # Get synthesis review
            review_data = await self.review_synthesis(
                compiled_sections, user_message, synthesis_outline, synthesis_model
            )
    
            # Apply edits from review
            await self.emit_synthesis_status("Applying improvements to synthesis...")
            edited_sections, changes_made = await self.apply_review_edits(
                compiled_sections, review_data, synthesis_model
            )
    
            # Format the bibliography
            bibliography_table = await self.format_bibliography_list(
                bibliography_data["bibliography"]
            )
    
            # Generate abstract
            await self.emit_synthesis_status("Generating abstract...")
            abstract = await self.generate_abstract(
                user_message,
                "".join(edited_sections.values()),
                bibliography_data["bibliography"],
            )
    
            # Build final answer
            comprehensive_answer = ""
    
            # Add title and subtitle
            main_title = titles.get("main_title", f"Research Report: {user_message}")
            subtitle = titles.get("subtitle", "A Comprehensive Analysis and Synthesis")
    
            comprehensive_answer += f"# {main_title}\n\n## {subtitle}\n\n"
    
            # Add abstract
            comprehensive_answer += f"## Abstract\n\n{abstract}\n\n"

            # Add introduction with compression
            await self.emit_synthesis_status("Generating introduction...")
            intro_prompt = {
                "role": "system",
                "content": PROMPTS["introduction_system"].format(query=user_message),
            }
    
            intro_context = f"Research Query: {user_message}\n\nResearch Outline:"
            for section in edited_sections:
                intro_context += f"\n- {section}"
    
            # Add compressed section content for better context
            section_context = "\n\nSection Content Summary:\n"
            for section_title, content in edited_sections.items():
                section_context += f"\n{section_title}: {content}...\n"
    
            # Compress the combined context
            combined_intro_context = intro_context + section_context
            intro_embedding = await self.get_embedding(combined_intro_context)
            compressed_intro_context = await self.compress_content_with_eigendecomposition(
                combined_intro_context, intro_embedding, None, None
            )
    
            intro_message = {"role": "user", "content": compressed_intro_context}
    
            try:
                # Use synthesis model for intro
                intro_response = await self.generate_completion(
                    synthesis_model,
                    [intro_prompt, intro_message],
                    stream=False,
                    temperature=self.valves.SYNTHESIS_TEMPERATURE * 0.83,
                    user_facing=True,
                )
    
                if (
                    intro_response
                    and "choices" in intro_response
                    and len(intro_response["choices"]) > 0
                ):
                    introduction = intro_response["choices"][0]["message"]["content"]
                    comprehensive_answer += f"## Introduction\n\n{introduction}\n\n"
                    await self.emit_synthesis_status("Introduction generation complete")
            except Exception as e:
                logger.error(f"Error generating introduction: {e}")
                comprehensive_answer += f"## Introduction\n\nThis research report addresses the query: '{user_message}'. The following sections present findings from a comprehensive investigation of this topic.\n\n"
                await self.emit_synthesis_status(
                    "Introduction generation failed, using fallback"
                )
    
            # Add each section with heading
            for section_title, content in edited_sections.items():
                # Get token count for the section
                memory_stats = self._ensure_memory_stats()
                section_tokens = memory_stats.get("section_tokens", {})
                section_tokens_count = section_tokens.get(section_title, 0)
                if section_tokens_count == 0:
                    section_tokens_count = await self.count_tokens(content)
                    section_tokens[section_title] = section_tokens_count
                    memory_stats["section_tokens"] = section_tokens
                    self.update_state("memory_stats", memory_stats)
    
                # Check for section title duplication in various formats
                if (
                    content.startswith(section_title)
                    or content.startswith(f"# {section_title}")
                    or content.startswith(f"## {section_title}")
                ):
                    # Remove first line and any following whitespace
                    content = (
                        content.split("\n", 1)[1].lstrip() if "\n" in content else content
                    )
    
                comprehensive_answer += f"## {section_title}\n\n{content}\n\n"
    
            # Add conclusion with compression
            await self.emit_synthesis_status("Generating conclusion...")
            concl_prompt = {
                "role": "system",
                "content": PROMPTS["conclusion_system"].format(query=user_message),
            }
    
            concl_context = (
                f"Research Query: {user_message}\n\nKey findings from each section:\n"
            )
    
            # Use compression for each section based on the model's context window
            full_content = ""
            for section_title, content in edited_sections.items():
                full_content += f"\n## {section_title}\n{content}\n\n"
    
            # Get embedding for compression context
            content_embedding = await self.get_embedding(full_content[:2000])
    
            # Apply intelligent compression based on your existing logic
            compressed_content = await self.compress_content_with_eigendecomposition(
                full_content,
                content_embedding,
                None,  # No summary embedding needed
                None,  # Let the compression function decide the ratio based on content
            )
    
            concl_context += compressed_content

            concl_message = {"role": "user", "content": concl_context}

            try:
                # Use synthesis model for conclusion
                concl_response = await self.generate_completion(
                    synthesis_model,
                    [concl_prompt, concl_message],
                    stream=False,
                    temperature=self.valves.SYNTHESIS_TEMPERATURE,
                    user_facing=True,
                )
    
                if (
                    concl_response
                    and "choices" in concl_response
                    and len(concl_response["choices"]) > 0
                ):
                    conclusion = concl_response["choices"][0]["message"]["content"]
                    comprehensive_answer += f"## Conclusion\n\n{conclusion}\n\n"
                    await self.emit_synthesis_status("Conclusion generation complete")
            except Exception as e:
                logger.error(f"Error generating conclusion: {e}")
                await self.emit_synthesis_status(
                    "Conclusion generation failed, using fallback"
                )

            # Link numeric in-text citations to bibliography anchors
            comprehensive_answer = self._link_numeric_citations_in_text(
                comprehensive_answer, bibliography_data["bibliography"]
            )
    
            # Add verification note if any citations were flagged
            comprehensive_answer = await self.add_verification_note(comprehensive_answer)
    
            # Add bibliography
            comprehensive_answer += f"{bibliography_table}\n\n"
    
            # Add research date
            comprehensive_answer += f"*Research conducted on: {self.research_date}*\n\n"
    
            # Count total tokens in the comprehensive answer
            synthesis_tokens = await self.count_tokens(comprehensive_answer)
            memory_stats = self._ensure_memory_stats()
            memory_stats["synthesis_tokens"] = synthesis_tokens
            self.update_state("memory_stats", memory_stats)
    
            # Calculate total tokens used in the research
            results_tokens = memory_stats.get("results_tokens", 0)
            section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
            total_tokens = results_tokens + section_tokens_sum + synthesis_tokens
            memory_stats["total_tokens"] = total_tokens
            self.update_state("memory_stats", memory_stats)

            # Mark research as completed
            self.update_state("research_completed", True)
            research_finished = True

            # Output the final compiled and edited synthesis
            await self.emit_synthesis_status("Final synthesis complete!", True)
    
            # Output the complete integrated synthesis
            await self.emit_message("\n\n## Comprehensive Answer\n\n")
            await self.emit_message(comprehensive_answer)
    
            # Add token usage statistics
            token_stats = (
                f"\n\n---\n\n**Token Usage Statistics**\n\n"
                f"- Research Results: {results_tokens} tokens\n"
                f"- Final Synthesis: {synthesis_tokens} tokens\n"
                f"- Total: {total_tokens} tokens\n"
            )
            await self.emit_message(token_stats)
    
            # Store the comprehensive answer for potential follow-up queries
            self.update_state("prev_comprehensive_summary", comprehensive_answer)
    
            # Share embedding cache stats
            cache_stats = self.embedding_cache.stats()
            logger.info(f"Embedding cache stats: {cache_stats}")
    
            # Export research data if enabled
            if self.valves.EXPORT_RESEARCH_DATA:
                try:
                    await self.emit_status("info", "Exporting research data...", False)
                    export_result = await self.export_research_data()
    
                    # Output information about where the export was saved
                    txt_filepath = export_result.get("txt_filepath", "")
                    # json_filepath = export_result.get("json_filepath", "")
    
                    export_message = (
                        f"\n\n---\n\n**Research Data Exported**\n\n"
                        f"Research data has been exported to:\n"
                        f"- Text file: `{txt_filepath}`\n\n"
                        f"This file contain all research results, queries, timestamps, and content for future reference."
                    )
    
                    await self.emit_message(export_message)
    
                except Exception as e:
                    logger.error(f"Error exporting research data: {e}")
                    await self.emit_message(
                        "*There was an error exporting the research data.*\n"
                    )

            # Complete the process
            await self.emit_status("success", "Deep research complete!", True)
            await self._cleanup_conversation_resources(conversation_id)
            cleanup_done = True
            return ""

        finally:
            if research_finished and not cleanup_done:
                await self._cleanup_conversation_resources(conversation_id)
                cleanup_done = True
            if not cleanup_done:
                await self._release_conversation_slot(conversation_id)
