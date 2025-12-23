from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeResearchStateMixin:
    async def initialize_research_state(
        self,
        user_message,
        research_outline,
        all_topics,
        outline_embedding,
        initial_results=None,
    ):
        """Initialize or reset research state consistently across interactive and non-interactive modes"""
        state = self.get_state()

        # Core research state
        self.update_state(
            "research_state",
            {
                "research_outline": research_outline,
                "all_topics": all_topics,
                "outline_embedding": outline_embedding,
                "user_message": user_message,
            },
        )

        # Initialize memory statistics with proper structure
        memory_stats = self._ensure_memory_stats()

        # Update results_tokens if we have initial results
        if initial_results:
            results_tokens = 0
            for result in initial_results:
                # Get or calculate tokens for this result
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                results_tokens += tokens

            # Update memory stats with token count
            memory_stats["results_tokens"] = results_tokens
            self.update_state("memory_stats", memory_stats)

        # Initialize tracking variables
        self.update_state("topic_usage_counts", state.get("topic_usage_counts", {}))
        self.update_state("completed_topics", state.get("completed_topics", set()))
        self.update_state("irrelevant_topics", state.get("irrelevant_topics", set()))
        self.update_state("active_outline", all_topics.copy())
        self.update_state("cycle_summaries", state.get("cycle_summaries", []))

        # Results tracking
        results_history = state.get("results_history", [])
        if initial_results:
            results_history.extend(initial_results)
        self.update_state("results_history", results_history)

        # Search history
        search_history = state.get("search_history", [])
        self.update_state("search_history", search_history)

        # Initialize dimension tracking
        await self.initialize_research_dimensions(all_topics, user_message)
        research_dimensions = state.get("research_dimensions")
        if research_dimensions:
            self.update_state(
                "latest_dimension_coverage", research_dimensions["coverage"].copy()
            )

        # Source tracking
        self.update_state("master_source_table", state.get("master_source_table", {}))
        self.update_state("url_selected_count", state.get("url_selected_count", {}))
        self.update_state("url_token_counts", state.get("url_token_counts", {}))

        # Trajectory accumulator reset
        self.trajectory_accumulator = None

        logger.info(
            f"Research state initialized with {len(all_topics)} topics and {len(results_history)} initial results"
        )

    async def update_token_counts(self, new_results=None):
        """Centralized function to update token counts consistently"""
        state = self.get_state()
        memory_stats = self._ensure_memory_stats()

        # Update results tokens if new results provided
        if new_results:
            for result in new_results:
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                memory_stats["results_tokens"] += tokens

        # If no results tokens but we have results history, recalculate
        results_history = state.get("results_history", [])
        if memory_stats["results_tokens"] == 0 and results_history:
            total_tokens = 0
            for result in results_history:
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                total_tokens += tokens
            memory_stats["results_tokens"] = total_tokens

        # Recalculate total tokens
        section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
        memory_stats["total_tokens"] = (
            memory_stats["results_tokens"]
            + section_tokens_sum
            + memory_stats.get("synthesis_tokens", 0)
        )

        # Update state
        self.update_state("memory_stats", memory_stats)

        return memory_stats

    async def update_topic_usage_counts(self, used_topics):
        """Update usage counts for topics used in queries"""
        state = self.get_state()
        topic_usage_counts = state.get("topic_usage_counts", {})

        # Increment counter for each used topic
        for topic in used_topics:
            topic_usage_counts[topic] = topic_usage_counts.get(topic, 0) + 1

        # Store updated counts
        self.update_state("topic_usage_counts", topic_usage_counts)
