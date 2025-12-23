from typing import Dict, Any, Set
import numpy as np

def default_memory_stats() -> Dict[str, Any]:
    return {
        "results_tokens": 0,
        "section_tokens": {},
        "synthesis_tokens": 0,
        "total_tokens": 0,
    }

def default_conversation_state() -> Dict[str, Any]:
    return {
        "research_completed": False,
        "prev_comprehensive_summary": "",
        "waiting_for_outline_feedback": False,
        "outline_feedback_data": None,
        "research_state": {},
        "follow_up_mode": False,
        "user_preferences": {
            "pdv": None,
            "strength": 0.0,
            "impact": 0.0,
            "labels": "",
        },
        "research_dimensions": None,
        "research_trajectory": None,
        "pdv_alignment_history": [],
        "gap_coverage_history": [],
        "semantic_transformations": None,
        "section_synthesized_content": {},
        "section_citations": {},
        "url_selected_count": {},
        "url_considered_count": {},
        "url_token_counts": {},
        "master_source_table": {},
        "global_citation_map": {},
        "verified_citations": [],
        "flagged_citations": [],
        "citation_fixes": [],
        "memory_stats": default_memory_stats(),
        "results_history": [],
        "search_history": [],
        "active_outline": [],
        "cycle_summaries": [],
        "completed_topics": set(),
        "irrelevant_topics": set(),
    }

class ResearchStateManager:
    """Manages research state per conversation to ensure proper isolation"""

    def __init__(self):
        self.conversation_states = {}

    def get_state(self, conversation_id):
        """Get state for a specific conversation, creating if needed"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = default_conversation_state()
        return self.conversation_states[conversation_id]

    def update_state(self, conversation_id, key, value):
        """Update a specific state value for a conversation"""
        state = self.get_state(conversation_id)
        state[key] = value

    def reset_state(self, conversation_id):
        """Reset the state for a specific conversation"""
        if conversation_id in self.conversation_states:
            del self.conversation_states[conversation_id]

class TrajectoryAccumulator:
    """Efficiently accumulates research trajectory across cycles"""

    def __init__(self, embedding_dim=1024):
        self.query_sum = np.zeros(embedding_dim)
        self.result_sum = np.zeros(embedding_dim)
        self.count = 0
        self.embedding_dim = embedding_dim

    def add_cycle_data(self, query_embeddings, result_embeddings, weight=1.0):
        """Add data from a research cycle"""
        if not query_embeddings or not result_embeddings:
            return

        # Simple averaging of embeddings
        query_centroid = np.mean(query_embeddings, axis=0)
        result_centroid = np.mean(result_embeddings, axis=0)

        # Add to accumulators with weight
        self.query_sum += query_centroid * weight
        self.result_sum += result_centroid * weight
        self.count += 1

    def get_trajectory(self):
        """Get the current trajectory vector"""
        if self.count == 0:
            return None

        query_centroid = self.query_sum / self.count
        result_centroid = self.result_sum / self.count
        trajectory = result_centroid - query_centroid

        norm = np.linalg.norm(trajectory)
        if norm > 1e-10:
            return (trajectory / norm).tolist()
        else:
            return None
