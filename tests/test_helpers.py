import asyncio
from types import SimpleNamespace

import pytest


def _pipe_with_context():
    import pipe as pipe_module

    pipe = pipe_module.Pipe()
    pipe.__user__ = SimpleNamespace(id="test-user")
    pipe._set_conversation_context("test-conv")
    return pipe


def test_chunk_text_sentence_level():
    pipe = _pipe_with_context()
    pipe.valves.CHUNK_LEVEL = 2
    chunks = pipe.chunk_text("First sentence. Second sentence! Third?")
    assert len(chunks) == 3
    assert chunks[0].startswith("First")


def test_extract_text_from_html_removes_nav():
    pipe = _pipe_with_context()
    html = "<html><nav>Menu</nav><body><p>Hello world</p></body></html>"
    text = asyncio.run(pipe.extract_text_from_html(html, prefer_bs4=False))
    assert "Hello world" in text
    assert "Menu" not in text


def test_compress_content_with_local_similarity(monkeypatch):
    pipe = _pipe_with_context()

    async def fake_get_embedding(text: str):
        # Simple deterministic 2D embedding based on length
        length = float(len(text))
        return [length, length % 7]

    pipe.get_embedding = fake_get_embedding  # type: ignore[assignment]
    content = (
        "Sentence one. Sentence two. Sentence three. Sentence four. "
        "Sentence five. Sentence six. Sentence seven. Sentence eight."
    )
    query_embedding = [1.0, 1.0]

    compressed = asyncio.run(
        pipe.compress_content_with_local_similarity(content, query_embedding, ratio=0.5)
    )

    assert compressed.count("Sentence") < content.count("Sentence")
    assert "Sentence" in compressed
