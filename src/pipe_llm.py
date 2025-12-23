from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeLLMMixin:
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        user_facing: bool = False,
    ):
        """Generate a completion from the specified model"""
        try:
            # Use provided temperature or default from valves
            if temperature is None:
                temperature = self.valves.TEMPERATURE

            # Enforce output language if configured
            # This hint is only applied to user-facing responses so that all internal
            # research prompts, queries, and planning remain in English.
            output_lang = getattr(self.valves, "OUTPUT_LANGUAGE", "auto") or "auto"
            output_lang = output_lang.lower()
            if user_facing and output_lang != "auto":
                if output_lang in ("pl", "polish"):
                    lang_hint = (
                        "Pisz wszystkie odpowiedzi wyłącznie po polsku, "
                        "nie używaj innych języków."
                    )
                elif output_lang in ("en", "english"):
                    lang_hint = (
                        "Write all responses only in English and do not "
                        "use other languages."
                    )
                else:
                    lang_hint = (
                        f"Write all responses only in {output_lang} and do not "
                        "use other languages."
                    )

                # Try to prepend language hint to the first system message
                system_found = False
                for msg in messages:
                    if msg.get("role") == "system":
                        original = msg.get("content", "")
                        msg["content"] = f"{lang_hint}\n\n{original}"
                        system_found = True
                        break

                # If no system message exists, add one
                if not system_found:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": lang_hint,
                        },
                    )

            debug_llm = getattr(self.valves, "DEBUG_LLM", False)
            if debug_llm:
                preview_messages = []
                for msg in messages:
                    preview_messages.append(
                        {
                            "role": msg.get("role"),
                            "content": truncate_for_log(msg.get("content", "")),
                        }
                    )
                logger.info(
                    "LLM DEBUG request model=%s temperature=%.3f stream=%s messages=%s",
                    model,
                    temperature,
                    stream,
                    preview_messages,
                )

            form_data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "keep_alive": "10m",
            }

            with self.timed(f"completion:{model}"):
                response = await generate_chat_completions(
                    self.__request__,
                    form_data,
                    user=self.__user__,
                )

            normalized = await self._normalize_completion_response(response)

            if debug_llm:
                # Try to extract the main assistant content for debugging
                try:
                    content = ""
                    if normalized and "choices" in normalized and normalized["choices"]:
                        content = normalized["choices"][0]["message"].get(
                            "content", ""
                        )
                    logger.info(
                        "LLM DEBUG response model=%s content=%s",
                        model,
                        truncate_for_log(content),
                    )
                except Exception as e:
                    logger.info(f"LLM DEBUG response logging failed: {e}")

            return normalized
        except Exception as e:
            logger.error(f"Error generating completion with model {model}: {e}")
            # Return a minimal valid response structure
            return {"choices": [{"message": {"content": f"Error: {str(e)}"}}]}

    async def _read_response_body(self, response: Any) -> bytes:
        if response is None:
            return b""

        # Streaming responses expose body_iterator
        iterator = getattr(response, "body_iterator", None)
        if iterator is not None:
            chunks: List[bytes] = []
            try:
                async for chunk in iterator:
                    chunks.append(self._chunk_to_bytes(chunk))
            except TypeError:
                for chunk in iterator:
                    chunks.append(self._chunk_to_bytes(chunk))
            return b"".join(chunks)

        # Standard responses often expose .body
        body = getattr(response, "body", None)
        if asyncio.iscoroutine(body):
            body = await body
        if body is not None:
            return self._chunk_to_bytes(body)

        # Fallback to .content
        content = getattr(response, "content", None)
        if asyncio.iscoroutine(content):
            content = await content
        if content is not None:
            return self._chunk_to_bytes(content)

        return self._chunk_to_bytes(response)

    async def _normalize_completion_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"choices": [{"message": {"content": response}}]}

        body_bytes = await self._read_response_body(response)

        # Close response objects if they expose a close coroutine/callable
        close_method = getattr(response, "close", None)
        if callable(close_method):
            try:
                close_result = close_method()
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception as close_error:
                logger.debug(f"Error closing completion response: {close_error}")

        if not body_bytes:
            return {"choices": [{"message": {"content": ""}}]}

        try:
            decoded = body_bytes.decode("utf-8", "replace")

            # Handle streaming-style "data: {...}" payloads by concatenating deltas.
            # For reasoning-capable models we:
            #   - accumulate visible answer from `content`
            #   - accumulate internal reasoning from `reasoning_content` separately
            #   - only surface `content` to callers (falling back to reasoning if no content at all)
            if decoded.strip().startswith("data:") or "data:" in decoded:
                content_chunks: List[str] = []
                reasoning_chunks: List[str] = []
                finish_reason = None
                try:
                    for line in decoded.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices") or []
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or choices[0].get("message") or {}

                        content_piece = delta.get("content")
                        if content_piece:
                            content_chunks.append(str(content_piece))

                        reasoning_piece = delta.get("reasoning_content")
                        if reasoning_piece:
                            reasoning_chunks.append(str(reasoning_piece))

                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

                    if content_chunks or reasoning_chunks:
                        visible_text = "".join(content_chunks) if content_chunks else "".join(
                            reasoning_chunks
                        )
                        message: Dict[str, Any] = {"content": visible_text}

                        # Preserve full reasoning for potential debugging without exposing
                        # it to normal flows that only read `message['content']`.
                        if reasoning_chunks:
                            message["reasoning_content"] = "".join(reasoning_chunks)

                        return {
                            "choices": [
                                {
                                    "message": message,
                                    "finish_reason": finish_reason,
                                }
                            ]
                        }
                except Exception as e:
                    logger.debug(f"Streaming response parse failed: {e}")

            # Non-streaming JSON payload
            return json.loads(decoded)
        except json.JSONDecodeError:
            decoded = body_bytes.decode("utf-8", "replace")
            return {"choices": [{"message": {"content": decoded}}]}

    async def emit_message(self, message: str):
        """Emit a message to the client"""
        try:
            await self.__current_event_emitter__(
                {"type": "message", "data": {"content": message}}
            )
        except Exception as e:
            logger.error(f"Error emitting message: {e}")

    async def emit_status(self, level: str, message: str, done: bool = False):
        """Emit a status message to the client"""
        try:
            # Check if research is completed
            state = self.get_state()
            research_completed = state.get("research_completed", False)

            if research_completed and not done:
                status = "complete"
            else:
                status = "complete" if done else "in_progress"

            await self.__current_event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error emitting status: {e}")

    async def emit_synthesis_status(self, message, is_done=False):
        """Emit both a status update and a chat message for synthesis progress"""
        await self.emit_status("info", message, is_done)
        await self.emit_message(f"*{message}*\n")

    def get_research_model(self):
        """Get the appropriate model for research/mechanical tasks"""
        # Always use the main research model
        return self.valves.RESEARCH_MODEL

    def get_synthesis_model(self):
        """Get the appropriate model for synthesis tasks"""
        if (
            self.valves.SYNTHESIS_MODEL
            and self.valves.SYNTHESIS_MODEL != self.valves.RESEARCH_MODEL
        ):
            return self.valves.SYNTHESIS_MODEL
        return self.valves.RESEARCH_MODEL

    def _chunk_to_bytes(self, chunk: Any) -> bytes:
        if chunk is None:
            return b""
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, str):
            return chunk.encode("utf-8", "replace")
        return str(chunk).encode("utf-8", "replace")
