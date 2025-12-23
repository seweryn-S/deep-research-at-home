from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeInteractiveMixin:
    async def rank_topics_by_research_priority(
        self,
        active_topics: List[str],
        gap_vector: Optional[List[float]] = None,
        completed_topics: Optional[Set[str]] = None,
        research_results: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Rank research topics by priority using semantic dimensions and gap analysis with dampening for frequently used topics"""
        if not active_topics:
            return []

        # If we only have a few topics, keep the original order
        if len(active_topics) <= 3:
            return active_topics

        # Get cache of topic alignments
        state = self.get_state()
        topic_alignment_cache = state.get("topic_alignment_cache", {})

        # Get topic usage counts for dampening
        topic_usage_counts = state.get("topic_usage_counts", {})
        dampening_factor = 0.9  # Each use reduces priority by 10%

        # Initialize scores for each topic
        topic_scores = {}

        # Get embeddings for all topics
        logger.info(f"Getting embeddings for {len(active_topics)} topics")
        topic_embeddings = {}

        # Get embeddings for each topic
        for topic in active_topics:
            embedding = await self.get_embedding(topic)
            if embedding:
                topic_embeddings[topic] = embedding

        # Get research trajectory for alignment calculation
        research_trajectory = state.get("research_trajectory")

        # Get user preferences
        user_preferences = state.get("user_preferences", {})
        pdv = user_preferences.get("pdv")
        pdv_impact = user_preferences.get("impact", 0.0)

        # Get current cycle for adaptive weights
        current_cycle = len(state.get("cycle_summaries", [])) + 1
        max_cycles = self.valves.MAX_CYCLES

        # Calculate weights for different factors based on research progress
        trajectory_weight = self.valves.TRAJECTORY_MOMENTUM

        # PDV weight calculation
        pdv_weight = 0.0
        if pdv is not None and pdv_impact > 0.1:
            pdv_alignment_history = state.get("pdv_alignment_history", [])
            if pdv_alignment_history:
                recent_alignment = sum(pdv_alignment_history[-3:]) / max(
                    1, len(pdv_alignment_history[-3:])
                )
                alignment_factor = min(1.0, recent_alignment * 2)
                pdv_weight = pdv_impact * alignment_factor

                # Apply adaptive fade-out
                fade_start_cycle = min(5, int(0.33 * max_cycles))
                if current_cycle > fade_start_cycle:
                    remaining_cycles = max_cycles - current_cycle
                    total_fade_cycles = max_cycles - fade_start_cycle
                    if total_fade_cycles > 0:
                        fade_ratio = remaining_cycles / total_fade_cycles
                        pdv_weight *= max(0.0, fade_ratio)
                    else:
                        pdv_weight = 0.0
            else:
                pdv_weight = pdv_impact

        # Gap weight calculation
        gap_weight = 0.0
        if gap_vector is not None:
            fade_start_cycle = min(5, int(0.5 * max_cycles))
            if current_cycle <= fade_start_cycle:
                gap_weight = self.valves.GAP_EXPLORATION_WEIGHT
            else:
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_ratio = remaining_cycles / total_fade_cycles
                    gap_weight = self.valves.GAP_EXPLORATION_WEIGHT * max(
                        0.0, fade_ratio
                    )

        # Content relevance weight increases over time
        relevance_weight = 0.2 + (0.3 * min(1.0, current_cycle / (max_cycles * 0.7)))

        # Normalize weights to sum to 1.0
        total_weight = trajectory_weight + pdv_weight + gap_weight + relevance_weight
        if total_weight > 0:
            trajectory_weight /= total_weight
            pdv_weight /= total_weight
            gap_weight /= total_weight
            relevance_weight /= total_weight

        logger.info(
            f"Priority weights: trajectory={trajectory_weight:.2f}, pdv={pdv_weight:.2f}, gap={gap_weight:.2f}, relevance={relevance_weight:.2f}"
        )

        # Prepare completed topics embeddings for relevance scoring
        completed_embeddings = {}
        if completed_topics and len(completed_topics) > 0 and relevance_weight > 0.0:
            # Limit number of completed topics to consider for efficiency
            completed_sample_size = min(10, len(completed_topics))
            completed_topics_list = list(completed_topics)[:completed_sample_size]

            # Get all completed topics embeddings sequentially
            completed_embed_results = []
            for topic in completed_topics_list:
                embedding = await self.get_embedding(topic)
                if embedding:
                    completed_embed_results.append(embedding)

            # Store valid embeddings with topic keys
            for i, embedding in enumerate(completed_embed_results):
                if embedding and i < len(completed_topics_list):
                    completed_embeddings[completed_topics_list[i]] = embedding

        # Prepare recent result embeddings for relevance scoring
        result_embeddings = {}
        if research_results and len(research_results) > 0 and relevance_weight > 0.0:
            # Get limited recent results (last 8 for efficiency)
            recent_results = research_results[-8:]

            # Prepare content for embedding
            result_contents = []
            for result in recent_results:
                content = result.get("content", "")[:2000]
                result_contents.append(content)

            # Get embeddings sequentially
            result_embed_results = []
            for content in result_contents:
                embedding = await self.get_embedding(content)
                if embedding:
                    result_embed_results.append(embedding)

            # Store valid embeddings with result index as key
            for i, embedding in enumerate(result_embed_results):
                if embedding and i < len(recent_results):
                    result_id = recent_results[i].get("url", "") or f"result_{i}"
                    result_embeddings[result_id] = embedding

        # Calculate scores for each topic
        for topic, topic_embedding in topic_embeddings.items():
            # Start with a base score
            score = 0.5
            component_scores = {}

            # Factor 1: Alignment with trajectory (research direction)
            if research_trajectory is not None and trajectory_weight > 0.0:
                # Check cache first
                cache_key = f"traj_{topic}"
                if cache_key in topic_alignment_cache:
                    traj_alignment = topic_alignment_cache[cache_key]
                else:
                    traj_alignment = np.dot(topic_embedding, research_trajectory)
                    # Normalize to 0-1 range
                    traj_alignment = (traj_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = traj_alignment

                component_scores["trajectory"] = traj_alignment * trajectory_weight

            # Factor 2: Alignment with user preference direction vector
            if pdv is not None and pdv_weight > 0.0:
                # Check cache first
                cache_key = f"pdv_{topic}"
                if cache_key in topic_alignment_cache:
                    pdv_alignment = topic_alignment_cache[cache_key]
                else:
                    pdv_alignment = np.dot(topic_embedding, pdv)
                    # Normalize to 0-1 range
                    pdv_alignment = (pdv_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = pdv_alignment

                component_scores["pdv"] = pdv_alignment * pdv_weight

            # Factor 3: Alignment with gap vector (unexplored areas)
            if gap_vector is not None and gap_weight > 0.0:
                # Check cache first
                cache_key = f"gap_{topic}"
                if cache_key in topic_alignment_cache:
                    gap_alignment = topic_alignment_cache[cache_key]
                else:
                    gap_alignment = np.dot(topic_embedding, gap_vector)
                    # Normalize to 0-1 range
                    gap_alignment = (gap_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = gap_alignment

                component_scores["gap"] = gap_alignment * gap_weight

            # Factor 4: Topic novelty compared to completed research
            if completed_embeddings and relevance_weight > 0.0:
                # Calculate average similarity to completed topics
                similarity_sum = 0
                count = 0

                for (
                    completed_topic,
                    completed_embedding,
                ) in completed_embeddings.items():
                    # Check cache first
                    cache_key = f"comp_{topic}_{completed_topic}"
                    if cache_key in topic_alignment_cache:
                        sim = topic_alignment_cache[cache_key]
                    else:
                        sim = cosine_similarity(
                            [topic_embedding], [completed_embedding]
                        )[0][0]
                        # Cache the result
                        topic_alignment_cache[cache_key] = sim

                    similarity_sum += sim
                    count += 1

                if count > 0:
                    avg_similarity = similarity_sum / count
                    # Invert - lower similarity means higher novelty
                    novelty = 1.0 - avg_similarity
                    component_scores["novelty"] = novelty * (relevance_weight * 0.5)

            # Factor 5: Information need based on search results
            if result_embeddings and relevance_weight > 0.0:
                # Calculate average relevance to results
                relevance_sum = 0
                count = 0

                for result_id, result_embedding in result_embeddings.items():
                    # Create cache key using result ID
                    cache_key = f"res_{topic}_{hash(result_id) % 10000}"

                    if cache_key in topic_alignment_cache:
                        rel = topic_alignment_cache[cache_key]
                    else:
                        rel = cosine_similarity([topic_embedding], [result_embedding])[
                            0
                        ][0]
                        # Cache the result
                        topic_alignment_cache[cache_key] = rel

                    relevance_sum += rel
                    count += 1

                if count > 0:
                    avg_relevance = relevance_sum / count
                    # Invert - lower relevance means higher information need
                    info_need = 1.0 - avg_relevance
                    component_scores["info_need"] = info_need * (relevance_weight * 0.5)

            # Calculate final score as sum of all component scores
            final_score = sum(component_scores.values())
            if not component_scores:
                final_score = 0.5  # Default if no components were calculated

            # Apply dampening based on usage count and result quality
            usage_count = topic_usage_counts.get(topic, 0)
            if usage_count > 0:
                # Get all results related to this topic
                topic_results = []

                # Look for results where the topic appears in the query or result content
                for result in research_results or []:
                    # Check if this result is relevant to this topic
                    result_content = result.get("content", "")[
                        :500
                    ]  # Use first 500 chars for efficiency
                    if topic in result.get("query", "") or topic in result_content:
                        topic_results.append(result)

                # If we have results for this topic, calculate quality-based dampening
                if topic_results:
                    # Calculate average similarity for this topic's results
                    avg_similarity = 0.0
                    count = 0
                    for result in topic_results:
                        similarity = result.get("similarity", 0.0)
                        if similarity > 0:  # Only count results with valid similarity
                            avg_similarity += similarity
                            count += 1

                    if count > 0:
                        avg_similarity /= count

                    # Scale dampening factor based on result quality
                    # similarity > 0.8: no penalty (dampening_multiplier = 1.0)
                    # similarity < 0.3: 50% penalty (dampening_multiplier = 0.5)
                    # Linear scaling between
                    if avg_similarity >= 0.8:
                        dampening_multiplier = 1.0
                    elif avg_similarity <= 0.3:
                        dampening_multiplier = 0.5
                    else:
                        # Linear scaling between 0.5 and 1.0
                        dampening_multiplier = 0.5 + (
                            0.5 * (avg_similarity - 0.3) / 0.5
                        )

                    logger.debug(
                        f"Topic '{topic}' quality-based dampening: {dampening_multiplier:.3f} (avg similarity: {avg_similarity:.3f}, from {count} results)"
                    )
                else:
                    # If no results yet, use the default dampening
                    dampening_multiplier = dampening_factor**usage_count
                    logger.debug(
                        f"Topic '{topic}' default dampening: {dampening_multiplier:.3f} (used {usage_count} times)"
                    )

                # Apply the dampening multiplier
                final_score *= dampening_multiplier

            # Store the score
            topic_scores[topic] = final_score

        # Update alignment cache with size limiting
        if len(topic_alignment_cache) > 300:  # Limit cache size
            # Create new cache with only recent entries
            new_cache = {}
            count = 0
            for k, v in reversed(list(topic_alignment_cache.items())):
                new_cache[k] = v
                count += 1
                if count >= 200:  # Keep 200 most recent entries
                    break
            topic_alignment_cache = new_cache

        self.update_state("topic_alignment_cache", topic_alignment_cache)

        # Sort topics by score (highest first)
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_topics = [topic for topic, score in sorted_topics]

        logger.info(f"Ranked {len(ranked_topics)} topics by research priority")
        return ranked_topics

    async def process_user_outline_feedback(
        self, outline_items: List[Dict], original_query: str
    ) -> Dict:
        """Process user feedback on research outline items by asking for feedback in chat"""
        # Number each outline item (maintain hierarchy but flatten for numbering)
        numbered_outline = []
        flat_items = []

        # Process the hierarchical outline structure
        item_num = 1
        for topic_item in outline_items:
            topic = topic_item.get("topic", "")
            subtopics = topic_item.get("subtopics", [])

            # Add main topic with number
            flat_items.append(topic)
            numbered_outline.append(f"{item_num}. {topic}")
            item_num += 1

            # Add subtopics with numbers
            for subtopic in subtopics:
                flat_items.append(subtopic)
                numbered_outline.append(f"{item_num}. {subtopic}")
                item_num += 1

        # Prepare the outline display
        outline_display = "\n".join(numbered_outline)

        # Choose language for user-facing instructions based on OUTPUT_LANGUAGE
        output_lang = (self.valves.OUTPUT_LANGUAGE or "auto").lower()
        is_pl = output_lang in ("pl", "polish")

        # Emit a message with instructions using improved slash commands
        if is_pl:
            feedback_message = (
                "### Plan badań\n\n"
                f"{outline_display}\n\n"
                "**Proszę o Twoją opinię na temat tego planu badań.**\n\n"
                "Możesz:\n"
                "- Użyć komend takich jak `/keep 1,3,5-7` lub `/remove 2,4,8-10`, aby wybrać konkretne pozycje po numerze\n"
                "- Albo po prostu opisać po polsku, na których tematach chcesz się skupić, a których wolisz uniknąć\n\n"
                "Przykłady:\n"
                "- `/k 1,3,5-7` (zachowaj tylko pozycje 1,3,5,6,7)\n"
                "- `/r 2,4,8-10` (usuń pozycje 2,4,8,9,10)\n"
                '- "Skup się na aspektach historycznych i pomiń szczegóły techniczne"\n'
                '- "Bardziej interesują mnie praktyczne zastosowania niż koncepcje teoretyczne"\n\n'
                "Jeśli chcesz kontynuować ze wszystkimi pozycjami, odpowiedz po prostu \"continue\" albo wyślij pustą wiadomość.\n\n"
                "**Zatrzymuję się tutaj i czekam na Twoją odpowiedź, zanim przejdziemy dalej z badaniami.**"
            )
        else:
            feedback_message = (
                "### Research Outline\n\n"
                f"{outline_display}\n\n"
                "**Please provide feedback on this research outline.**\n\n"
                "You can:\n"
                "- Use commands like `/keep 1,3,5-7` or `/remove 2,4,8-10` to select specific items by number\n"
                "- Or simply describe what topics you want to focus on or avoid in natural language\n\n"
                "Examples:\n"
                "- `/k 1,3,5-7` (keep only items 1,3,5,6,7)\n"
                "- `/r 2,4,8-10` (remove items 2,4,8,9,10)\n"
                "- \"Focus on historical aspects and avoid technical details\"\n"
                "- \"I'm more interested in practical applications than theoretical concepts\"\n\n"
                "If you want to continue with all items, just reply 'continue' or leave your message empty.\n\n"
                "**I'll pause here to await your response before continuing the research.**"
            )

        await self.emit_message(feedback_message)

        # Set flag to indicate we're waiting for feedback
        self.update_state("waiting_for_outline_feedback", True)
        self.update_state(
            "outline_feedback_data",
            {
                "outline_items": outline_items,
                "flat_items": flat_items,
                "numbered_outline": numbered_outline,
                "original_query": original_query,
            },
        )

        # Return a default response (this will be overridden in the next call)
        return {
            "kept_items": flat_items,
            "removed_items": [],
            "kept_indices": list(range(len(flat_items))),
            "removed_indices": [],
            "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        }

    async def process_natural_language_feedback(
        self, user_message: str, flat_items: List[str]
    ) -> Dict:
        """Process natural language feedback to determine which topics to keep/remove"""

        # Create a prompt for the model to interpret user feedback
        interpret_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant analyzing user feedback on a research outline.
	Based on the user's natural language input, determine which research topics should be kept or removed.
	
	The user's message expresses preferences about the research direction. Analyze this to identify:
	1. Which specific topics from the outline align with their interests
	2. Which specific topics should be removed based on their preferences
	
	Your task is to categorize each topic as EITHER "keep" OR "remove", NEVER both, based on the user's natural language feedback.
    Don't allow your own biases or preferences to have any affect on your answer - please remain purely objective and user research-oriented.
	Provide your response as a JSON object with two lists: "keep" for indices to keep, and "remove" for indices to remove.
	Indices should be 0-based (first item is index 0).""",
        }

        # Prepare context with list of topics and user message
        topics_list = "\n".join([f"{i}. {topic}" for i, topic in enumerate(flat_items)])

        context = f"""Research outline topics:
	{topics_list}
	
	User feedback:
	"{user_message}"
	
	Based on this feedback, categorize each topic (by index) as either "keep" or "remove".
	If the user clearly expresses a preference to focus on certain topics or avoid others, use that to guide your decisions.
	If the user's feedback is ambiguous about some topics, categorize them based on their similarity to clearly mentioned preferences.
	"""

        # Generate interpretation of user feedback
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [interpret_prompt, {"role": "user", "content": context}],
                temperature=self.valves.TEMPERATURE
                * 0.3,  # Low temperature for consistent interpretation
            )

            result_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                json_str = result_content[
                    result_content.find("{") : result_content.rfind("}") + 1
                ]
                result_data = json.loads(json_str)

                # Get keep and remove lists
                keep_indices = result_data.get("keep", [])
                remove_indices = result_data.get("remove", [])

                # Ensure both keep_indices and remove_indices are lists
                if not isinstance(keep_indices, list):
                    keep_indices = []
                if not isinstance(remove_indices, list):
                    remove_indices = []

                # Ensure each index is in either keep or remove
                all_indices = set(range(len(flat_items)))
                missing_indices = all_indices - set(keep_indices) - set(remove_indices)

                # By default, keep missing indices
                keep_indices.extend(missing_indices)

                # Convert to kept and removed items
                kept_items = [
                    flat_items[i] for i in keep_indices if i < len(flat_items)
                ]
                removed_items = [
                    flat_items[i] for i in remove_indices if i < len(flat_items)
                ]

                logger.info(
                    f"Natural language feedback interpretation: keep {len(kept_items)}, remove {len(removed_items)}"
                )

                return {
                    "kept_items": kept_items,
                    "removed_items": removed_items,
                    "kept_indices": keep_indices,
                    "removed_indices": remove_indices,
                }

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing feedback interpretation: {e}")
                # Default to keeping all items
                return {
                    "kept_items": flat_items,
                    "removed_items": [],
                    "kept_indices": list(range(len(flat_items))),
                    "removed_indices": [],
                }

        except Exception as e:
            logger.error(f"Error interpreting natural language feedback: {e}")
            # Default to keeping all items
            return {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
            }

    async def process_outline_feedback_continuation(self, user_message: str):
        """Process the user feedback received in a continuation call"""
        # Get the data from the previous call
        state = self.get_state()
        feedback_data = state.get("outline_feedback_data", {})
        outline_items = feedback_data.get("outline_items", [])
        flat_items = feedback_data.get("flat_items", [])
        original_query = feedback_data.get("original_query", "")

        # Process the user input
        user_input = user_message.strip()

        # If user just wants to continue with all items
        if user_input.lower() == "continue" or not user_input:
            await self.emit_message(
                "\n*Continuing with all research outline items.*\n\n"
            )
            return {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
                "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
            }

        # Check if it's a slash command (keep or remove)
        slash_keep_patterns = [r"^/k\s", r"^/keep\s"]
        slash_remove_patterns = [r"^/r\s", r"^/remove\s"]

        is_keep_cmd = any(
            re.match(pattern, user_input) for pattern in slash_keep_patterns
        )
        is_remove_cmd = any(
            re.match(pattern, user_input) for pattern in slash_remove_patterns
        )

        # Process slash commands
        if is_keep_cmd or is_remove_cmd:
            # Extract the item indices/ranges part
            if is_keep_cmd:
                items_part = re.sub(r"^(/k|/keep)\s+", "", user_input).replace(",", " ")
            else:
                items_part = re.sub(r"^(/r|/remove)\s+", "", user_input).replace(
                    ",", " "
                )

            # Process the indices and ranges
            selected_indices = set()
            for part in items_part.split():
                part = part.strip()
                if not part:
                    continue

                # Check if it's a range (e.g., 5-9)
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-"))
                        # Validate range bounds before converting to 0-indexed
                        if (
                            start < 1
                            or start > len(flat_items)
                            or end < 1
                            or end > len(flat_items)
                        ):
                            await self.emit_message(
                                f"Invalid range '{part}': valid range is 1-{len(flat_items)}. Skipping."
                            )
                            continue

                        # Convert to 0-indexed
                        start = start - 1
                        end = end - 1
                        selected_indices.update(range(start, end + 1))
                    except ValueError:
                        await self.emit_message(
                            f"Invalid range format: '{part}'. Skipping."
                        )
                else:
                    # Single number
                    try:
                        idx = int(part)
                        # Validate index before converting to 0-indexed
                        if idx < 1 or idx > len(flat_items):
                            await self.emit_message(
                                f"Index {idx} out of range: valid range is 1-{len(flat_items)}. Skipping."
                            )
                            continue

                        # Convert to 0-indexed
                        idx = idx - 1
                        selected_indices.add(idx)
                    except ValueError:
                        await self.emit_message(f"Invalid number: '{part}'. Skipping.")

            # Convert to lists
            selected_indices = sorted(list(selected_indices))

            # Determine kept and removed indices based on mode
            if is_keep_cmd:
                # Keep mode - selected indices are kept, others removed
                kept_indices = selected_indices
                removed_indices = [
                    i for i in range(len(flat_items)) if i not in kept_indices
                ]
            else:
                # Remove mode - selected indices are removed, others kept
                removed_indices = selected_indices
                kept_indices = [
                    i for i in range(len(flat_items)) if i not in removed_indices
                ]

            # Get the actual items
            kept_items = [flat_items[i] for i in kept_indices if i < len(flat_items)]
            removed_items = [
                flat_items[i] for i in removed_indices if i < len(flat_items)
            ]
        else:
            # Process natural language feedback
            nl_feedback = await self.process_natural_language_feedback(
                user_input, flat_items
            )

            # Make sure we have a valid response, not None
            if nl_feedback is None:
                # Default to keeping all items
                nl_feedback = {
                    "kept_items": flat_items,
                    "removed_items": [],
                    "kept_indices": list(range(len(flat_items))),
                    "removed_indices": [],
                }

            kept_items = nl_feedback.get("kept_items", flat_items)
            removed_items = nl_feedback.get("removed_items", [])
            kept_indices = nl_feedback.get("kept_indices", list(range(len(flat_items))))
            removed_indices = nl_feedback.get("removed_indices", [])

        # Calculate preference direction vector based on kept and removed items
        preference_vector = await self.calculate_preference_direction_vector(
            kept_items, removed_items, flat_items
        )

        # Update user_preferences in state with the new preference vector
        self.update_state("user_preferences", preference_vector)
        logger.info(
            f"Updated user_preferences with PDV impact: {preference_vector.get('impact', 0.0):.3f}"
        )

        # Show the user what's happening
        await self.emit_message("\n### Feedback Processed\n")

        if kept_items:
            kept_list = "\n".join([f"✓ {item}" for item in kept_items])
            await self.emit_message(
                f"**Keeping {len(kept_items)} items:**\n{kept_list}\n"
            )

        if removed_items:
            removed_list = "\n".join([f"✗ {item}" for item in removed_items])
            await self.emit_message(
                f"**Removing {len(removed_items)} items:**\n{removed_list}\n"
            )

        await self.emit_message("Generating replacement items for removed topics...\n")

        return {
            "kept_items": kept_items,
            "removed_items": removed_items,
            "kept_indices": kept_indices,
            "removed_indices": removed_indices,
            "preference_vector": preference_vector,
        }

    async def group_replacement_topics(self, replacement_topics):
        """Group replacement topics semantically into groups of 2-4 topics each"""
        # Skip if too few topics
        if len(replacement_topics) <= 4:
            return [replacement_topics]  # Just one group if 4 or fewer topics

        # Get embeddings for each topic sequentially
        topic_embeddings = []
        for i, topic in enumerate(replacement_topics):
            embedding = await self.get_embedding(topic)
            if embedding:
                topic_embeddings.append((topic, embedding))

        # If we don't have enough valid embeddings for grouping, use simple groups
        if len(topic_embeddings) < 3:
            logger.warning(
                "Not enough embeddings for semantic grouping, using simple groups"
            )
            # Just divide topics into groups of 4
            groups = []
            for i in range(0, len(replacement_topics), 4):
                groups.append(replacement_topics[i : i + 4])
            return groups

        try:
            # Extract embeddings into a numpy array
            embeddings_array = np.array([emb for _, emb in topic_embeddings])

            # Determine number of clusters (groups)
            total_topics = len(topic_embeddings)
            # Aim for groups of 3-4 topics each
            n_clusters = max(1, total_topics // 3)
            # Cap at a reasonable number
            n_clusters = min(n_clusters, 5)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings_array)

            # Group topics by cluster
            grouped_topics = {}
            for i, (topic, _) in enumerate(topic_embeddings):
                cluster_id = kmeans.labels_[i]
                if cluster_id not in grouped_topics:
                    grouped_topics[cluster_id] = []
                grouped_topics[cluster_id].append(topic)

            # Get the groups as a list
            groups_list = list(grouped_topics.values())

            # Balance any groups that are too small or large
            if len(groups_list) > 1:
                # Sort groups by size
                groups_list.sort(key=len)

                # Merge any tiny groups (fewer than 2 topics)
                while len(groups_list) > 1 and len(groups_list[0]) < 2:
                    smallest = groups_list.pop(0)
                    second_smallest = groups_list[0]  # Don't remove yet, just reference

                    # Merge with second smallest
                    groups_list[0] = second_smallest + smallest

                    # Re-sort
                    groups_list.sort(key=len)

                # Split any very large groups (more than 5 topics)
                for i, group in enumerate(groups_list):
                    if len(group) > 5:
                        # Simple split at midpoint
                        midpoint = len(group) // 2
                        groups_list[i] = group[:midpoint]  # First half
                        groups_list.append(group[midpoint:])  # Second half

            return groups_list

        except Exception as e:
            logger.error(f"Error during topic grouping: {e}")
            # Fall back to simple grouping on error
            groups = []
            for i in range(0, len(replacement_topics), 4):
                groups.append(replacement_topics[i : i + 4])
            return groups

    async def generate_group_query(self, topic_group, user_message):
        """Generate a search query that covers a group of related topics"""
        if not topic_group:
            return user_message

        topics_text = ", ".join(topic_group)

        # Create a prompt for generating the query
        prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating an effective search query. 
	Create a search query that will find relevant information for a group of related topics aimed at addressing the original user input.
	The query should be specific enough to find targeted information while broadly representing all topics in the group.
	Make the query concise (maximum 10 words) and focused.""",
        }

        # Create the message content
        message = {
            "role": "user",
            "content": f"""Generate a search query for this group of topics:
	{topics_text}
	
	This is related to the original user query: "{user_message}"
	
	Generate a single concise search query that will find information relevant to these topics.
	Just respond with the search query text only.""",
        }

        # Generate the query
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [prompt, message],
                temperature=self.valves.TEMPERATURE * 0.7,
            )

            query = response["choices"][0]["message"]["content"].strip()

            # Clean up the query: remove quotes and ensure it's not too long
            query = query.replace('"', "").replace('"', "").replace('"', "")

            # If the query is too long, truncate it
            if len(query.split()) > 12:
                query = " ".join(query.split()[:12])

            return query

        except Exception as e:
            logger.error(f"Error generating group query: {e}")
            # Fallback: combine the first topic with the user message
            return f"{user_message} {topic_group[0]}"

    async def extract_topic_relevant_info(self, results, topics):
        """Extract information from search results specifically relevant to given topics"""
        if not results:
            return []

        # Create a prompt for extracting relevant information
        extraction_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant extracting information from search results.
	Identify and extract information that is specifically relevant to the given topics.
	Format the extracted information as concise bullet points, focusing on facts, data, and insights.
	Ignore general information not directly related to the topics.""",
        }

        # Create context with search results and topics
        topics_str = ", ".join(topics)
        extraction_context = f"Topics: {topics_str}\n\nSearch Results:\n\n"

        for i, result in enumerate(results):
            extraction_context += f"Result {i+1}:\n"
            extraction_context += f"Title: {result.get('title', 'Untitled')}\n"
            extraction_context += f"Content: {result.get('content', '')}...\n\n"

        extraction_context += "\nExtract relevant information for the listed topics from these search results."

        # Create messages for extraction
        extraction_messages = [
            extraction_prompt,
            {"role": "user", "content": extraction_context},
        ]

        # Extract relevant information
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                extraction_messages,
                temperature=self.valves.TEMPERATURE
                * 0.4,  # Lower temperature for factual extraction
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                extracted_info = response["choices"][0]["message"]["content"]
                return extracted_info
            else:
                return "No relevant information found."

        except Exception as e:
            logger.error(f"Error extracting topic-relevant information: {e}")
            return "Error extracting information from search results."

    async def refine_topics_with_research(
        self, topics, relevant_info, pdv, original_query
    ):
        """Refine topics based on both user preferences and research results"""
        # Create a prompt for refining topics
        refine_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant refining research topics.
	Based on the extracted information and user preferences, revise each topic to:
	1. Be specific and targeted based on the research findings, while maintaining alignment with user preferences and the original query
	2. Prioritize topics that seem most relevant to answering the query and that will reasonably result in worthwhile expanded research
	3. Be phrased as clear, researchable topics in the same style as those to be replaced
	
	Your refined topics should incorporate new discoveries that heighten and expand upon the intent of the original query.
    Avoid overstating the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
    You do not need to include justification along with your refined topics.""",
        }

        # Create context with topics, research info, and preference direction
        pdv_context = ""
        if pdv is not None:
            pdv_context = "\nUser preferences are directing research toward topics similar to what was kept and away from what was removed."

        refine_context = f"""Original topics: {', '.join(topics)}
	
	Original query: {original_query}
	
	Extracted research information:
	{relevant_info}
	{pdv_context}
	
	Refine these topics based on the research findings and user preferences.
	Provide a list of the same number of refined topics."""

        # Create messages for refinement
        refine_messages = [refine_prompt, {"role": "user", "content": refine_context}]

        # Generate refined topics
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                refine_messages,
                temperature=self.valves.TEMPERATURE
                * 0.7,  # Balanced temperature for creativity with focus
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                refined_content = response["choices"][0]["message"]["content"]

                # Extract topics using regex (looking for numbered or bulleted list items)
                refined_topics = re.findall(
                    r"(?:^|\n)(?:\d+\.\s*|\*\s*|-\s*)([^\n]+)", refined_content
                )

                # If we couldn't extract enough topics, use the original ones
                if len(refined_topics) < len(topics):
                    logger.warning(
                        f"Not enough refined topics extracted ({len(refined_topics)}), using originals"
                    )
                    return topics

                # Limit to the same number as original topics
                refined_topics = refined_topics[: len(topics)]
                return refined_topics
            else:
                return topics

        except Exception as e:
            logger.error(f"Error refining topics: {e}")
            return topics

    async def continue_research_after_feedback(
        self,
        feedback_result,
        user_message,
        outline_items,
        all_topics,
        outline_embedding,
    ):
        """Continue the research process after receiving user feedback on the outline"""
        kept_items = feedback_result["kept_items"]
        removed_items = feedback_result["removed_items"]
        preference_vector = feedback_result["preference_vector"]

        # If there are no removed items, skip the replacement logic and return original outline
        if not removed_items:
            await self.emit_message(
                "\n*No changes made to research outline. Continuing with original outline.*\n\n"
            )
            self.update_state(
                "research_state",
                {
                    "research_outline": outline_items,
                    "all_topics": all_topics,
                    "outline_embedding": outline_embedding,
                    "user_message": user_message,
                },
            )

            # Clear waiting flag
            self.update_state("waiting_for_outline_feedback", False)
            return outline_items, all_topics, outline_embedding

        # Generate replacement topics for removed items if needed
        if removed_items:
            await self.emit_status("info", "Generating replacement topics...", False)
            replacement_topics = await self.generate_replacement_topics(
                user_message,
                kept_items,
                removed_items,
                preference_vector,
                all_topics,
            )

            if replacement_topics:
                # Group replacement topics semantically
                topic_groups = await self.group_replacement_topics(replacement_topics)

                # Get state for tracking URLs
                state = self.get_state()
                url_selected_count = state.get("url_selected_count", {})

                # Get initial results to track URLs from previous cycles
                results_history = state.get("results_history", [])

                # Create a set of already seen URLs from all previous research
                previously_seen_urls = set()
                for result in results_history:
                    url = result.get("url", "")
                    if url:
                        previously_seen_urls.add(url)

                # Also track URLs we see during this replacement cycle
                replacement_cycle_seen_urls = set()

                # For each group, generate and execute targeted queries
                group_results = []
                for group in topic_groups:
                    # Generate a query that covers this group of topics
                    group_query = await self.generate_group_query(group, user_message)

                    # Get query embedding
                    query_embedding = await self.get_embedding(group_query)

                    # Execute search for this group
                    await self.emit_message(
                        f"**Researching topics:** {', '.join(group)}\n**Query:** {group_query}\n\n"
                    )
                    results = await self.process_query(
                        group_query, query_embedding, outline_embedding
                    )

                    # Filter out URLs we've seen in previous cycles or this replacement cycle
                    filtered_results = []
                    for result in results:
                        url = result.get("url", "")

                        # Skip if we've seen this URL in previous cycles or this replacement cycle
                        if url and (
                            url in previously_seen_urls
                            or url in replacement_cycle_seen_urls
                        ):
                            continue

                        # Keep new URLs we haven't seen before
                        filtered_results.append(result)
                        if url:
                            replacement_cycle_seen_urls.add(
                                url
                            )  # Mark as seen in this cycle

                    # If we have no results after filtering but had some initially, use fallback
                    if not filtered_results and results:
                        # Use a fallback approach - find the least seen URL
                        least_seen = None
                        min_seen_count = float("inf")

                        for result in results:
                            url = result.get("url", "")
                            seen_count = url_selected_count.get(url, 0)

                            if seen_count < min_seen_count:
                                min_seen_count = seen_count
                                least_seen = result

                        if least_seen:
                            filtered_results.append(least_seen)
                            if least_seen.get("url"):
                                replacement_cycle_seen_urls.add(least_seen.get("url"))
                            logger.info(
                                f"Using least-seen URL as fallback to ensure research continues"
                            )

                    group_results.append(
                        {
                            "topics": group,
                            "query": group_query,
                            "results": filtered_results,
                        }
                    )

                # Now refine each topic based on both PDV and search results
                refined_topics = []
                for group in group_results:
                    topics = group["topics"]
                    results = group["results"]

                    # Extract key information from results relevant to these topics
                    relevant_info = await self.extract_topic_relevant_info(
                        results, topics
                    )

                    # Generate refined topics that incorporate both user preferences and new research
                    refined = await self.refine_topics_with_research(
                        topics,
                        relevant_info,
                        self.get_state().get("user_preferences", {}).get("pdv"),
                        user_message,
                    )

                    refined_topics.extend(refined)

                # Use these refined topics in place of the original replacement topics
                replacement_topics = refined_topics

                # Create new research outline structure
                new_research_outline = []
                new_all_topics = []

                # Track the original hierarchy
                original_hierarchy = {}  # Store parent-child relationships
                original_main_topics = set()  # Track which items were main topics
                original_subtopics = set()  # Track which items were subtopics

                # Extract from the original outline structure
                for topic_item in outline_items:
                    topic = topic_item["topic"]
                    original_main_topics.add(topic)
                    subtopics = topic_item.get("subtopics", [])

                    # Track the hierarchy
                    for subtopic in subtopics:
                        original_hierarchy[subtopic] = topic
                        original_subtopics.add(subtopic)

                # Process kept items to maintain hierarchy
                for topic_item in outline_items:
                    topic = topic_item["topic"]
                    subtopics = topic_item.get("subtopics", [])

                    if topic in kept_items:
                        # Keep the original topic with its kept subtopics
                        kept_subtopics = [s for s in subtopics if s in kept_items]
                        if kept_subtopics:  # Only add if there are kept subtopics
                            new_topic_item = {
                                "topic": topic,
                                "subtopics": kept_subtopics,
                            }
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(topic)
                            new_all_topics.extend(kept_subtopics)
                        else:
                            # If main topic is kept but no subtopics, still add it
                            new_topic_item = {"topic": topic, "subtopics": []}
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(topic)
                    else:
                        # For removed main topics, check if any subtopics were kept
                        kept_subtopics = [s for s in subtopics if s in kept_items]
                        if kept_subtopics:
                            # Just restore the original main topic name teehee
                            revised_topic = f"{topic}"
                            new_topic_item = {
                                "topic": revised_topic,
                                "subtopics": kept_subtopics,
                            }
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(revised_topic)
                            new_all_topics.extend(kept_subtopics)

                # Process orphaned kept items (not already added)
                orphaned_kept_items = [
                    item for item in kept_items if item not in new_all_topics
                ]

                # Get embeddings for assignment
                if orphaned_kept_items and new_research_outline:
                    try:
                        # Try to add orphaned items to existing topics based on semantic similarity
                        main_topic_embeddings = {}
                        for outline_item in new_research_outline:
                            topic = outline_item["topic"]
                            embedding = await self.get_embedding(topic)
                            if embedding:
                                main_topic_embeddings[topic] = embedding

                        for item in orphaned_kept_items:
                            item_embedding = await self.get_embedding(item)
                            if item_embedding:
                                # Find best match
                                best_match = None
                                best_score = 0.5  # Threshold

                                for (
                                    topic,
                                    topic_embedding,
                                ) in main_topic_embeddings.items():
                                    similarity = cosine_similarity(
                                        [item_embedding], [topic_embedding]
                                    )[0][0]
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_match = topic

                                if best_match:
                                    # Add to existing topic
                                    for outline_item in new_research_outline:
                                        if outline_item["topic"] == best_match:
                                            outline_item["subtopics"].append(item)
                                            new_all_topics.append(item)
                                            break
                                else:
                                    # If no good match, create a new topic from the item
                                    if item in original_main_topics:
                                        # It was a main topic, keep it that way
                                        new_research_outline.append(
                                            {"topic": item, "subtopics": []}
                                        )
                                        new_all_topics.append(item)
                                    else:
                                        # It was a subtopic, but now it's orphaned, make it a main topic
                                        new_research_outline.append(
                                            {"topic": item, "subtopics": []}
                                        )
                                        new_all_topics.append(item)
                            else:
                                # No embedding, add as a main topic
                                new_research_outline.append(
                                    {"topic": item, "subtopics": []}
                                )
                                new_all_topics.append(item)
                    except Exception as e:
                        logger.error(f"Error assigning orphaned items: {e}")
                        # Add all orphaned items as main topics on error
                        for item in orphaned_kept_items:
                            new_research_outline.append(
                                {"topic": item, "subtopics": []}
                            )
                            new_all_topics.append(item)
                elif orphaned_kept_items:
                    # No existing topics to add to, make each orphaned item a main topic
                    for item in orphaned_kept_items:
                        new_research_outline.append({"topic": item, "subtopics": []})
                        new_all_topics.append(item)

                # Add replacement topics now
                # First, try to add them to semantically similar existing main topics
                if replacement_topics and new_research_outline:
                    try:
                        # Get embeddings for existing main topics
                        main_topic_embeddings = {}
                        for outline_item in new_research_outline:
                            topic = outline_item["topic"]
                            embedding = await self.get_embedding(topic)
                            if embedding:
                                main_topic_embeddings[topic] = embedding

                        # Track which replacements have been assigned
                        assigned_replacements = set()

                        # Try to assign each replacement to a semantically similar main topic
                        for replacement in replacement_topics:
                            replacement_embedding = await self.get_embedding(
                                replacement
                            )
                            if replacement_embedding:
                                # Find best match
                                best_match = None
                                best_score = 0.65  # Higher threshold for replacements

                                for (
                                    topic,
                                    topic_embedding,
                                ) in main_topic_embeddings.items():
                                    similarity = cosine_similarity(
                                        [replacement_embedding], [topic_embedding]
                                    )[0][0]
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_match = topic

                                if best_match:
                                    # Add to existing topic
                                    for outline_item in new_research_outline:
                                        if outline_item["topic"] == best_match:
                                            outline_item["subtopics"].append(
                                                replacement
                                            )
                                            new_all_topics.append(replacement)
                                            assigned_replacements.add(replacement)
                                            break

                        # Create new topics for unassigned replacements
                        unassigned_replacements = [
                            r
                            for r in replacement_topics
                            if r not in assigned_replacements
                        ]

                        # Group the unassigned replacements
                        replacement_groups = await self.group_replacement_topics(
                            unassigned_replacements
                        )

                        for group in replacement_groups:
                            # Generate title for the group
                            try:
                                group_title = await self.generate_group_title(
                                    group, user_message
                                )
                            except Exception as e:
                                logger.error(f"Error generating group title: {e}")
                                group_title = f"Additional Research Area {len(new_research_outline) - len(outline_items) + 1}"

                            # Add as a new main topic
                            new_research_outline.append(
                                {"topic": group_title, "subtopics": group}
                            )
                            new_all_topics.append(group_title)
                            new_all_topics.extend(group)

                    except Exception as e:
                        logger.error(f"Error during replacement topic assignment: {e}")
                        # Fallback: add all replacements as a new group
                        group_title = "Additional Research Topics"
                        new_research_outline.append(
                            {"topic": group_title, "subtopics": replacement_topics}
                        )
                        new_all_topics.append(group_title)
                        new_all_topics.extend(replacement_topics)
                elif replacement_topics:
                    # No existing outline to add to, create groups from replacements
                    replacement_groups = await self.group_replacement_topics(
                        replacement_topics
                    )

                    for i, group in enumerate(replacement_groups):
                        try:
                            group_title = await self.generate_group_title(
                                group, user_message
                            )
                        except Exception as e:
                            logger.error(f"Error generating group title: {e}")
                            group_title = f"Research Group {i+1}"

                        new_research_outline.append(
                            {"topic": group_title, "subtopics": group}
                        )
                        new_all_topics.append(group_title)
                        new_all_topics.extend(group)

                # Update the research outline and topic list
                if new_research_outline:  # Only update if we have valid content
                    research_outline = new_research_outline
                    all_topics = new_all_topics

                    # Update outline embedding based on all_topics
                    outline_text = " ".join(all_topics)
                    outline_embedding = await self.get_embedding(outline_text)

                    # Re-initialize dimension tracking with new topics
                    await self.initialize_research_dimensions(all_topics, user_message)

                    # Make sure to store initial coverage for later display
                    research_dimensions = state.get("research_dimensions")
                    if research_dimensions:
                        # Make a copy to avoid reference issues
                        self.update_state(
                            "latest_dimension_coverage",
                            research_dimensions["coverage"].copy(),
                        )
                        logger.info(
                            f"Updated dimension coverage after feedback with {len(research_dimensions['coverage'])} values"
                        )

                        # Also update trajectory accumulator for consistency
                        self.trajectory_accumulator = (
                            None  # Reset for fresh accumulation
                        )

                    # Show the updated outline to the user
                    updated_outline = "### Updated Research Outline\n\n"
                    for topic_item in research_outline:
                        updated_outline += f"**{topic_item['topic']}**\n"
                        for subtopic in topic_item.get("subtopics", []):
                            updated_outline += f"- {subtopic}\n"
                        updated_outline += "\n"

                    await self.emit_message(updated_outline)

                    # Updated message about continuing with main research
                    await self.emit_message(
                        "\n*Updated research outline with user preferences. Continuing to main research cycles...*\n\n"
                    )

                    # Store the updated research state
                    self.update_state(
                        "research_state",
                        {
                            "research_outline": research_outline,
                            "all_topics": all_topics,
                            "outline_embedding": outline_embedding,
                            "user_message": user_message,
                        },
                    )

                    # Clear waiting flag
                    self.update_state("waiting_for_outline_feedback", False)
                    return research_outline, all_topics, outline_embedding
                else:
                    # If we couldn't create a valid outline, continue with original
                    await self.emit_message(
                        "\n*No valid outline could be created. Continuing with original outline.*\n\n"
                    )
                    self.update_state(
                        "research_state",
                        {
                            "research_outline": outline_items,
                            "all_topics": all_topics,
                            "outline_embedding": outline_embedding,
                            "user_message": user_message,
                        },
                    )

                    # Clear waiting flag
                    self.update_state("waiting_for_outline_feedback", False)
                    return outline_items, all_topics, outline_embedding
            else:
                # No items were removed, continue with original outline
                await self.emit_message(
                    "\n*No changes made to research outline. Continuing with original outline.*\n\n"
                )
                self.update_state(
                    "research_state",
                    {
                        "research_outline": outline_items,
                        "all_topics": all_topics,
                        "outline_embedding": outline_embedding,
                        "user_message": user_message,
                    },
                )

                # Clear waiting flag
                self.update_state("waiting_for_outline_feedback", False)
                return outline_items, all_topics, outline_embedding

    async def generate_group_title(self, topics: List[str], user_message: str) -> str:
        """Generate a descriptive title for a group of related topics"""
        if not topics:
            return ""

        # For very small groups, just combine the topics
        if len(topics) <= 2:
            combined = " & ".join(topics)
            if len(combined) > 80:
                return combined[:77] + "..."
            return combined

        # Create a prompt to generate the group title
        title_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant creating a concise descriptive title for a group of related research topics.
    Create a short, clear title (4-8 words) that captures the common theme across these topics.
    The title should be specific enough to distinguish this group from others, but general enough to encompass all topics.
    DO NOT use generic phrases like "Research Group" or "Topic Group".
    Respond with ONLY the title text.""",
        }

        # Create the message content with full topics
        topic_text = "\n- " + "\n- ".join(topics)

        message = {
            "role": "user",
            "content": f"""Create a concise title for this group of related research topics:
    {topic_text}
    
    These topics are part of research about: "{user_message}"
    
    Respond with ONLY the title (4-8 words).""",
        }

        # Generate the title
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [title_prompt, message],
                temperature=0.7,
                user_facing=True,
            )

            title = response["choices"][0]["message"]["content"].strip()

            # Remove quotes if present
            title = title.strip("\"'")

            # Limit length if needed
            if len(title) > 80:
                title = title[:77] + "..."

            return title
        except Exception as e:
            logger.error(f"Error generating group title: {e}")
            # Single clean fallback that uses first topic
            return f"{topics[0][:40]}... & Related Topics"

    async def is_follow_up_query(self, messages: List[Dict]) -> bool:
        """Determine if the current query is a follow-up to a previous research session"""
        # If we have a previous comprehensive summary and research has been completed,
        # treat any new query as a follow-up
        state = self.get_state()
        prev_comprehensive_summary = state.get("prev_comprehensive_summary", "")
        research_completed = state.get("research_completed", False)

        # Check if we're waiting for outline feedback - if so, don't treat as new or follow-up
        waiting_for_outline_feedback = state.get("waiting_for_outline_feedback", False)
        if waiting_for_outline_feedback:
            return False

        # Check for fresh conversation by examining message count
        # A brand new conversation will have very few messages
        is_new_conversation = (
            len(messages) <= 2
        )  # Only 1-2 messages in a new conversation

        # If this appears to be a new conversation and we're not waiting for feedback,
        # don't treat as follow-up and reset state
        if is_new_conversation and not waiting_for_outline_feedback:
            # Reset the state for this conversation to ensure clean start
            self.reset_state()
            return False

        return bool(prev_comprehensive_summary and research_completed)

    async def generate_replacement_topics(
        self,
        query: str,
        kept_items: List[str],
        removed_items: List[str],
        preference_vector: Dict,
        outline_items: List[str],
    ) -> List[str]:
        """Generate replacement topics using semantic transformation"""
        # If nothing was removed, return empty list
        if not removed_items:
            return []

        # If nothing was kept, use the full original outline as reference
        if not kept_items:
            kept_items = outline_items

        # Calculate 80% of removed items count, rounded up
        num_replacements = math.ceil(len(removed_items) * 0.8)

        # Ensure at least one replacement
        num_replacements = max(1, num_replacements)

        logger.info(
            f"Generating {num_replacements} replacement topics (80% of {len(removed_items)} removed)"
        )

        # Create a prompt to generate replacements
        replacement_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating replacement topics for a research outline.
    Based on the kept topics, original query, and user's preferences, generate new research topics to replace removed ones.
    Each new topic should:
    1. Be directly relevant to answering or addressing the original query
    2. Be conceptually aligned with the kept topics
    3. Avoid concepts related to removed topics and their associated themes
    4. Be specific and actionable for research without devolving into hyperspecificity
    
    Generate EXACTLY the requested number of replacement topics in a numbered list format.
    Each replacement should be thoughtful and unique, exploring and expanding on different aspects of the research subject.
    """,
        }

        # Extract preference information
        pdv = preference_vector.get("pdv")
        strength = preference_vector.get("strength", 0.0)
        impact = preference_vector.get("impact", 0.0)

        # Prepare the request content
        content = f"""Original query: {query}
    
    Kept topics (conceptually preferred):
    {kept_items}
    
    Removed topics (to avoid):
    {removed_items}
    
    """

        # Pre-compute embeddings
        state = self.get_state()
        if pdv is not None and impact > 0.1:
            # Get query embedding first
            query_embedding = await self.get_embedding(query)

            # Get kept item embeddings sequentially
            kept_embeddings = []
            for item in kept_items:
                embedding = await self.get_embedding(item)
                if embedding:
                    kept_embeddings.append(embedding)

            # If we have enough embeddings, create a semantic transformation
            if query_embedding and len(kept_embeddings) >= 3:
                # Create a simple eigendecomposition
                try:
                    # Filter out any non-array elements that could cause errors
                    valid_embeddings = []
                    for emb in kept_embeddings:
                        if isinstance(emb, list) or (
                            hasattr(emb, "ndim") and emb.ndim == 1
                        ):
                            valid_embeddings.append(emb)

                    # Only proceed if we have enough valid embeddings
                    if len(valid_embeddings) >= 3:
                        kept_array = np.array(valid_embeddings)
                        # Simple PCA
                        pca = PCA(n_components=min(3, len(valid_embeddings)))
                        pca.fit(kept_array)
                    else:
                        logger.warning(
                            f"Not enough valid embeddings for PCA: {len(valid_embeddings)}/3 required"
                        )
                        return []

                    eigen_data = {
                        "eigenvectors": pca.components_.tolist(),
                        "eigenvalues": pca.explained_variance_.tolist(),
                        "explained_variance": pca.explained_variance_ratio_.tolist(),
                    }

                    # Create transformation that includes PDV
                    transformation = await self.create_semantic_transformation(
                        eigen_data, pdv=pdv
                    )

                    # Store for later use
                    self.update_state("semantic_transformations", transformation)

                    logger.info(
                        f"Created semantic transformation for replacement topics generation"
                    )
                except Exception as e:
                    logger.error(f"Error creating PCA for topic replacement: {e}")

        if pdv is not None:
            # Translate the PDV into natural language concepts
            pdv_concepts = await self.translate_pdv_to_words(pdv)
            if pdv_concepts:
                content += f"User preferences: The user prefers topics related to: {pdv_concepts}\n"
                if strength > 0.9:
                    content += f"The user has expressed a strong preference for these concepts. "
                elif strength > 0.5:
                    content += f"The user has expressed a moderate preference for these concepts. "
                else:
                    content += f"The user has expressed a slight preference for these concepts. "

        content += f"""Generate EXACTLY {num_replacements} replacement research topics in a numbered list.
    These should align with the kept topics and original query, while avoiding concepts from removed topics.
    Please don't include any other text in your response but the replacement topics. You don't need to justify them either.
    """

        messages = [replacement_prompt, {"role": "user", "content": content}]

        # Generate all replacements at once
        try:
            await self.emit_status(
                "info", f"Generating {num_replacements} replacement topics...", False
            )

            # Generate replacements
            # Use research model for generating replacements
            research_model = self.get_research_model()
            response = await self.generate_completion(
                research_model,
                messages,
                temperature=self.valves.TEMPERATURE
                * 1.1,  # Slightly higher temperature for creative replacements
                user_facing=True,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]

                # Parse the generated text to extract topics (numbered list format)
                lines = generated_text.split("\n")
                replacements = []

                for line in lines:
                    # Look for numbered list items: 1. Topic description
                    match = re.search(r"^\s*\d+\.\s*(.+)$", line)
                    if match:
                        topic = match.group(1).strip()
                        if (
                            topic and len(topic) > 10
                        ):  # Minimum length to be a valid topic
                            replacements.append(topic)

                # Ensure we have exactly the right number of replacements
                if len(replacements) > num_replacements:
                    replacements = replacements[:num_replacements]
                elif len(replacements) < num_replacements:
                    # If we didn't get enough, create generic ones to fill the gap
                    while len(replacements) < num_replacements:
                        missing_count = num_replacements - len(replacements)
                        await self.emit_status(
                            "info",
                            f"Generating {missing_count} additional topics...",
                            False,
                        )
                        replacements.append(
                            f"Additional research on {query} aspect {len(replacements)+1}"
                        )

                return replacements

        except Exception as e:
            logger.error(f"Error generating replacement topics: {e}")

        # Fallback - create generic replacements
        return [
            f"Alternative research topic {i+1} for {query}"
            for i in range(num_replacements)
        ]

    async def improved_query_generation(
        self, user_message, priority_topics, search_context
    ):
        """Generate refined search queries for research topics with improved context"""
        query_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating effective search queries.
    Based on the user's original question, current research needs, and context provided, generate 4 precise search queries.
    Each query should be specific, use relevant keywords, and be designed to find targeted information.
    
    Your queries should:
    1. Directly address the priority research topics
    2. Avoid redundancy with previous queries
    3. Target information gaps in the current research
    4. Be concise (6-12 words) but specific 
    5. Include specialized terminology when appropriate
    
    Focus on core conceptual terms with targeted expansions and don't return heavy, clunky queries.
    Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.
    
    Format your response as a valid JSON object with the following structure:
    {"queries": [
      "query": "search query 1", "topic": "related research topic", 
      "query": "search query 2", "topic": "related research topic",
      "query": "search query 3", "topic": "related research topic",
      "query": "search query 4", "topic": "related research topic"
    ]}""",
        }

        message = {
            "role": "user",
            "content": f"""Original query: "{user_message}"\n\nResearch context: "{search_context}"\n\nGenerate 4 effective search queries to gather information for the priority research topics.""",
        }

        # Generate the queries first, without any embedding operations
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [query_prompt, message],
                temperature=self.valves.TEMPERATURE,
            )

            query_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                query_json_str = query_content[
                    query_content.find("{") : query_content.rfind("}") + 1
                ]
                query_data = json.loads(query_json_str)
                queries = query_data.get("queries", [])

                # Check if queries is a list of strings or a list of objects
                if queries and isinstance(queries[0], str):
                    # Convert to objects with query and topic
                    query_strings = queries
                    query_topics = (
                        priority_topics[: len(queries)]
                        if priority_topics
                        else ["Research"] * len(queries)
                    )
                    queries = [
                        {"query": q, "topic": t}
                        for q, t in zip(query_strings, query_topics)
                    ]

                return queries

            except Exception as e:
                logger.error(f"Error parsing query JSON: {e}")
                # Fallback: generate basic queries for priority topics
                queries = []
                for i, topic in enumerate(priority_topics[:3]):
                    queries.append({"query": f"{user_message} {topic}", "topic": topic})

                return queries

        except Exception as e:
            logger.error(f"Error generating improved queries: {e}")
            # Fallback: generate basic queries
            queries = []
            for i, topic in enumerate(priority_topics[:3]):
                queries.append({"query": f"{user_message} {topic}", "topic": topic})

            return queries
