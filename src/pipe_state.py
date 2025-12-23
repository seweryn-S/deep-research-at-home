import asyncio
import concurrent.futures
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from src.state import default_memory_stats

logger = logging.getLogger("Deep Research at Home")


class PipeStateMixin:
    def _set_conversation_context(self, conversation_id: str):
        """Bind the current task to a conversation ID."""
        self.conversation_id = conversation_id
        try:
            self._conversation_ctx.set(conversation_id)
        except Exception:
            pass

    def _resolve_conversation_id(self, body: dict, user_id: str) -> str:
        """Resolve a stable conversation ID for the current OpenWebUI chat."""
        candidate_sources = []
        if isinstance(body, dict):
            candidate_sources.extend(
                [
                    ("body.chat_id", body.get("chat_id")),
                    ("body.chatId", body.get("chatId")),
                    ("body.conversation_id", body.get("conversation_id")),
                    ("body.conversationId", body.get("conversationId")),
                ]
            )
            chat_block = body.get("chat")
            if isinstance(chat_block, dict):
                candidate_sources.extend(
                    [
                        ("body.chat.id", chat_block.get("id")),
                        ("body.chat.chat_id", chat_block.get("chat_id")),
                        ("body.chat.chatId", chat_block.get("chatId")),
                    ]
                )
            messages = body.get("messages", [])
            if isinstance(messages, list) and messages:
                first_message = messages[0]
                if isinstance(first_message, dict):
                    candidate_sources.extend(
                        [
                            ("messages[0].chat_id", first_message.get("chat_id")),
                            ("messages[0].chatId", first_message.get("chatId")),
                        ]
                    )

        for label, candidate in candidate_sources:
            if isinstance(candidate, str) and candidate.strip():
                conversation_id = f"{user_id}_{candidate.strip()}"
                if getattr(self.valves, "DEBUG_SEARCH", False):
                    logger.info(
                        "CONVERSATION DEBUG: conversation_id=%r source=%s raw=%r",
                        conversation_id,
                        label,
                        candidate,
                    )
                return conversation_id

        messages = body.get("messages", []) if isinstance(body, dict) else []
        first_message = messages[0] if messages else {}
        message_id = (
            first_message.get("id", "default")
            if isinstance(first_message, dict)
            else "default"
        )
        conversation_id = f"{user_id}_{message_id}"
        if getattr(self.valves, "DEBUG_SEARCH", False):
            source = "messages[0].id" if message_id != "default" else "default"
            logger.info(
                "CONVERSATION DEBUG: conversation_id=%r source=%s raw=%r",
                conversation_id,
                source,
                message_id,
            )
        return conversation_id

    def _get_current_conversation_id(self) -> str:
        """Retrieve the conversation ID bound to this task."""
        conversation_id = self._conversation_ctx.get(None)
        if conversation_id:
            return conversation_id
        if self.conversation_id:
            return self.conversation_id

        user_id = getattr(getattr(self, "__user__", None), "id", "anonymous")
        temp_id = f"temp_{hash(str(user_id))}"
        self._set_conversation_context(temp_id)
        return temp_id

    def get_state(self):
        """Get the current conversation state."""
        conversation_id = self._get_current_conversation_id()
        return self.state_manager.get_state(conversation_id)

    def update_state(self, key, value):
        """Update a specific state value."""
        conversation_id = self._get_current_conversation_id()
        self.state_manager.update_state(conversation_id, key, value)

    def reset_state(self, conversation_id: Optional[str] = None):
        """Reset the state for the current conversation."""
        conv_id = conversation_id or self._get_current_conversation_id()
        if conv_id:
            self.state_manager.reset_state(conv_id)
            self.trajectory_accumulator = None
            self.is_pdf_content = False
            logger.info("Full state reset for conversation: %s", conv_id)

    @contextmanager
    def timed(self, label: str):
        """Log elapsed time when DEBUG_TIMING is enabled."""
        start = time.perf_counter()
        try:
            yield
        finally:
            if getattr(self.valves, "DEBUG_TIMING", False):
                elapsed = time.perf_counter() - start
                logger.info("TIMING %s: %.3fs", label, elapsed)

    def debug_log(self, message: str, *args):
        """Log debug info only when DEBUG_LLM is enabled."""
        if getattr(self.valves, "DEBUG_LLM", False):
            logger.debug(message, *args)

    def _ensure_memory_stats(self) -> Dict[str, Any]:
        """Guarantee memory_stats exists and is shaped correctly."""
        state = self.get_state()
        memory_stats = state.get("memory_stats")
        if not isinstance(memory_stats, dict):
            memory_stats = default_memory_stats()
            self.update_state("memory_stats", memory_stats)
            return memory_stats

        defaults = default_memory_stats()
        for key, default_value in defaults.items():
            if key not in memory_stats:
                memory_stats[key] = (
                    default_value
                    if not isinstance(default_value, dict)
                    else default_value.copy()
                )
        self.update_state("memory_stats", memory_stats)
        return memory_stats

    def _ensure_tracking_maps(self):
        """Ensure common tracking dictionaries exist."""
        required_maps = [
            "master_source_table",
            "url_selected_count",
            "url_considered_count",
            "url_token_counts",
        ]
        for key in required_maps:
            state = self.get_state()
            if key not in state or state.get(key) is None:
                self.update_state(key, {})

    def _max_conversations(self) -> int:
        """Number of allowed concurrent sessions."""
        try:
            limit = int(getattr(self.valves, "PARALLEL_SESSIONS", 2))
            return max(1, min(limit, 10))
        except Exception:
            return 2

    def _active_count(self) -> Tuple[int, int]:
        """Current active count and limit."""
        cls = type(self)
        return len(cls._active_conversations), self._max_conversations()

    async def _acquire_conversation_slot(self, conversation_id: str) -> bool:
        """Ensure we respect the maximum number of concurrent conversations."""
        wait_notified = False

        while True:
            acquired = False
            limit = self._max_conversations()
            cls = type(self)
            async with cls._conversation_lock:
                if conversation_id in cls._active_conversations or len(
                    cls._active_conversations
                ) < limit:
                    cls._active_conversations.add(conversation_id)
                    logger.info(
                        "Conversation %s acquired slot (%d/%d)",
                        conversation_id,
                        len(cls._active_conversations),
                        limit,
                    )
                    acquired = True

            if acquired:
                if wait_notified:
                    await self.emit_message(
                        "*Wolny slot dostępny – rozpoczynamy badanie. "
                        f"Aktywne sesje: {len(type(self)._active_conversations)}/{limit}*\n"
                    )
                return True

            if not wait_notified:
                active, limit = self._active_count()
                await self.emit_status(
                    "info",
                    f"Trwa maksymalna liczba badań ({active}/{limit}). Twoje zapytanie czeka na wolny slot.",
                    False,
                )
                await self.emit_message(
                    "*Dodano do kolejki. Rozpoczniemy badanie, gdy zwolni się miejsce. "
                    f"Aktualnie aktywne: {active}/{limit}*\n"
                )
                wait_notified = True

            await asyncio.sleep(1.0)

    async def _release_conversation_slot(self, conversation_id: str):
        """Release a previously acquired conversation slot."""
        cls = type(self)
        async with cls._conversation_lock:
            if conversation_id in cls._active_conversations:
                cls._active_conversations.remove(conversation_id)
                active, limit = self._active_count()
                logger.info(
                    "Conversation %s released slot (%d/%d)", conversation_id, active, limit
                )

    def _ensure_executor(self, conversation_id: str):
        """Get or create an executor dedicated to this conversation."""
        executor = self._conversation_executors.get(conversation_id)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.valves.THREAD_WORKERS
            )
            self._conversation_executors[conversation_id] = executor
        return executor

    async def _cleanup_conversation_resources(self, conversation_id: str):
        """Tear down per-conversation resources and caches."""
        try:
            executor = self._conversation_executors.pop(conversation_id, None)
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.warning("Error shutting down executor for %s: %s", conversation_id, e)

        try:
            self.state_manager.reset_state(conversation_id)
        except Exception as e:
            logger.warning("Error resetting state for %s: %s", conversation_id, e)

        await self._release_conversation_slot(conversation_id)

        try:
            self._conversation_ctx.set(None)
        except Exception:
            pass

