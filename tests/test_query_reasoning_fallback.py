"""Request-local fallback through real Agents/LiteLLM serialization, offline only."""

import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from test_codex_stream_transport import sse_turn


@pytest_asyncio.fixture
async def query_wire(monkeypatch, tmp_path):
    import litellm
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    from litellm.llms.custom_httpx import llm_http_handler
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.main import acompletion

    state = SimpleNamespace(requests=[], replies=[], auth_reads=[])
    monkeypatch.setenv("HOME", str(tmp_path))

    def forbidden_auth(*args, **kwargs):
        state.auth_reads.append(True)
        raise AssertionError("OAuth file access forbidden")

    monkeypatch.setattr(Authenticator, "__init__", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_access_token", lambda self: "offline-token")
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_api_base", lambda self: "https://codex.invalid")
    monkeypatch.setattr(Authenticator, "_read_auth_file", forbidden_auth)
    monkeypatch.setattr(ChatGPTResponsesAPIConfig, "validate_environment", lambda *a, **k: {})
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "num_retries", 0)

    async def capture(request):
        state.requests.append(json.loads(request.content))
        assert state.replies, "Unexpected extra leaf request"
        status, body = state.replies.pop(0)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": body}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **k: client)
    try:
        yield state
    finally:
        await client.client.aclose()


UNSUPPORTED = "Unsupported value: 'none' for reasoning.effort"


@pytest.mark.asyncio
@pytest.mark.parametrize("bundled", [False, True])
async def test_only_current_turn_retries_without_replaying_tool(
    query_wire, kb_dir, monkeypatch, bundled, caplog
):
    import logging

    from agents import RunConfig

    from openkb.agent import query
    from openkb.config import LlmCredentialBundle

    monkeypatch.setattr(query, "get_agents_md", lambda path: "Fixture schema")
    monkeypatch.setattr(
        "openkb.config.resolve_effective_config",
        lambda path: ({"query_reasoning_effort": "none"}, {}),
    )
    tool_calls = []

    def read(path, root):
        tool_calls.append(path)
        return "Fixture wiki contents"

    monkeypatch.setattr(query, "read_wiki_file", read)
    query_wire.replies.extend(
        [
            (200, sse_turn("Read wiki.", tool=True)),
            (400, UNSUPPORTED),
            (200, sse_turn("Fixture answer.")),
        ]
    )
    bundle = LlmCredentialBundle(reasoning_effort="none") if bundled else None
    run_config = query.build_run_config_from_bundle("chatgpt/gpt-6-luna", bundle)
    if run_config is None:
        run_config = RunConfig(tracing_disabled=True)
    else:
        run_config.tracing_disabled = True
    with caplog.at_level(logging.WARNING):
        answer = await query.run_query(
            "Read index", kb_dir, "chatgpt/gpt-6-luna", bundle=bundle, run_config=run_config
        )
    assert answer == "Fixture answer."
    assert tool_calls == ["index.md"]
    assert [r["reasoning"]["effort"] for r in query_wire.requests] == ["none", "none", "low"]
    assert {r["model"] for r in query_wire.requests} == {"gpt-6-luna"}
    assert query_wire.requests[1]["input"] == query_wire.requests[2]["input"]
    assert any(i["type"] == "function_call_output" for i in query_wire.requests[1]["input"])
    assert not query_wire.auth_reads
    assert "requested=none effective=low fallback=true scope=current_request" in caplog.text
    if bundled:
        assert bundle.reasoning_effort == "none"
        assert run_config.model_settings.reasoning.effort == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_tool", [False, True])
async def test_streaming_never_retries_even_before_first_event(
    query_wire, kb_dir, monkeypatch, after_tool
):
    from agents import RunConfig

    from openkb.agent import query

    calls = []
    monkeypatch.setattr(query, "read_wiki_file", lambda *args: calls.append(args) or "Wiki")
    if after_tool:
        query_wire.replies.append((200, sse_turn("Checking wiki.", tool=True)))
    query_wire.replies.append((400, UNSUPPORTED))
    agent = query.build_query_agent(
        str(kb_dir / "wiki"), "chatgpt/gpt-6-luna", reasoning_effort="none"
    )
    events = []
    with pytest.raises(Exception, match="Unsupported value"):
        async for event in query.iter_agent_response_events(
            agent, "Read index", run_config=RunConfig(tracing_disabled=True)
        ):
            events.append(event)
    assert len(query_wire.requests) == (2 if after_tool else 1)
    assert all(r["reasoning"]["effort"] == "none" for r in query_wire.requests)
    assert len(calls) == (1 if after_tool else 0)
    assert any(e["event"] == "delta" for e in events) == after_tool
    assert not query_wire.auth_reads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effort,status,message",
    [
        ("none", 400, "Malformed request"),
        ("none", 400, UNSUPPORTED + " quota"),
        ("none", 401, UNSUPPORTED),
        ("none", 403, UNSUPPORTED),
        ("none", 429, UNSUPPORTED),
        ("medium", 400, UNSUPPORTED),
        ("high", 400, UNSUPPORTED),
        (None, 400, UNSUPPORTED),
    ],
)
async def test_noneligible_errors_never_retry(query_wire, kb_dir, effort, status, message):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent

    query_wire.replies.append((status, message))
    agent = build_query_agent(str(kb_dir / "wiki"), "chatgpt/gpt-6-luna", reasoning_effort=effort)
    with pytest.raises(Exception):
        await Runner.run(agent, "Fixture", run_config=RunConfig(tracing_disabled=True))
    assert len(query_wire.requests) == 1
    if effort is not None:
        assert query_wire.requests[0]["reasoning"]["effort"] == effort
    assert not query_wire.auth_reads
    if effort is None:
        assert agent.model == "litellm/chatgpt/gpt-6-luna"


@pytest.mark.asyncio
async def test_failed_low_attempt_is_terminal(query_wire, kb_dir):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent

    query_wire.replies.extend([(400, UNSUPPORTED), (400, UNSUPPORTED)])
    agent = build_query_agent(str(kb_dir / "wiki"), "chatgpt/gpt-6-luna", reasoning_effort="none")
    with pytest.raises(Exception, match="Unsupported value"):
        await Runner.run(agent, "Fixture", run_config=RunConfig(tracing_disabled=True))
    assert [r["reasoning"]["effort"] for r in query_wire.requests] == ["none", "low"]
    assert agent.model_settings.reasoning.effort == "none"
    assert agent.model_settings.extra_args == {"allowed_openai_params": ["reasoning_effort"]}


@pytest.mark.asyncio
async def test_fallback_does_not_change_next_request_or_shared_settings(query_wire, kb_dir):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent

    query_wire.replies.extend(
        [
            (400, UNSUPPORTED),
            (200, sse_turn("First fixture.")),
            (200, sse_turn("Second fixture.")),
        ]
    )
    agent = build_query_agent(str(kb_dir / "wiki"), "chatgpt/gpt-6-luna", reasoning_effort="none")
    settings = agent.model_settings
    reasoning = settings.reasoning
    extra = settings.extra_args
    config = RunConfig(tracing_disabled=True)
    first = await Runner.run(agent, "First", run_config=config)
    second = await Runner.run(agent, "Second", run_config=config)
    assert (first.final_output, second.final_output) == ("First fixture.", "Second fixture.")
    assert [r["reasoning"]["effort"] for r in query_wire.requests] == ["none", "low", "none"]
    assert agent.model_settings is settings
    assert settings.reasoning is reasoning
    assert settings.extra_args is extra
    assert reasoning.effort == "none"
