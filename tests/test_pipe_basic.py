import asyncio

import types


def test_pipe_has_required_interface():
    import pipe as pipe_module

    pipe = pipe_module.Pipe()

    # Minimal conversation/user context so get_state works
    pipe.__user__ = types.SimpleNamespace(id="test-user")
    pipe._set_conversation_context("test-conv")

    # Minimal conversation/user context so get_state works
    pipe.__user__ = types.SimpleNamespace(id="test-user")
    pipe._set_conversation_context("test-conv")

    # Minimal conversation/user context so get_state works
    pipe.__user__ = types.SimpleNamespace(id="test-user")
    pipe._set_conversation_context("test-conv")

    # Required attributes for Open WebUI pipes
    assert hasattr(pipe, "valves")
    assert hasattr(pipe, "type")
    assert callable(getattr(pipe, "pipes"))
    assert callable(getattr(pipe, "pipe"))

    pipes_list = pipe.pipes()
    assert isinstance(pipes_list, list)
    assert pipes_list, "pipes() should return at least one definition"
    assert "id" in pipes_list[0]
    assert "name" in pipes_list[0]


def test_search_client_delegation(monkeypatch):
    import pipe as pipe_module

    pipe = pipe_module.Pipe()

    # Replace low-level methods with simple stubs
    async def fake_try(_self, query: str):
        return [{"title": "t1", "url": "u1", "snippet": "s1"}]

    async def fake_fallback(_self, query: str):
        return [{"title": "t2", "url": "u2", "snippet": "s2"}]

    pipe.search_client._try_openwebui_search = types.MethodType(fake_try, pipe.search_client)
    pipe.search_client._fallback_search = types.MethodType(fake_fallback, pipe.search_client)

    async def run():
        pipe.__user__ = types.SimpleNamespace(id="test-user")
        pipe._set_conversation_context("test-conv")
        results = await pipe.search_client.search_web("test query")
        assert isinstance(results, list)
        assert results[0]["title"] == "t1"

    asyncio.run(run())
