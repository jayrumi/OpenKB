"""Offline wire-level regressions using the pinned SDKs, never OAuth or inference."""

import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio


class RequestCaptured(BaseException):
    """Stop at HTTP transport without substituting a final model result."""


@pytest_asyncio.fixture
async def wire(monkeypatch):
    import litellm
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    from litellm.llms.custom_httpx import llm_http_handler
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.main import acompletion

    state = SimpleNamespace(requests=[], replies=[])

    async def capture(request):
        state.requests.append(request)
        if not state.replies:
            raise RequestCaptured
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=state.replies.pop(0)
        )

    # The guard blocks inference; selectively enable the real entry point ONLY
    # behind an HTTP MockTransport and replace auth before any provider lookup.
    monkeypatch.setattr(Authenticator, "__init__", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_access_token", lambda self: "offline-token")
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_api_base", lambda self: "https://codex.invalid")

    def forbidden_auth(*args, **kwargs):
        raise AssertionError("OAuth file access forbidden")

    monkeypatch.setattr(Authenticator, "_read_auth_file", forbidden_auth)
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(ChatGPTResponsesAPIConfig, "validate_environment", lambda *a, **k: {})
    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **k: client)
    try:
        yield state
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6.2-sol", "gpt-6-luna"])
@pytest.mark.parametrize("stream", [True, False])
async def test_query_reaches_codex_with_stream_true(wire, kb_dir, model, stream):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent, iter_agent_response_events

    agent = build_query_agent(str(kb_dir / "wiki"), f"chatgpt/{model}")
    try:
        if stream:
            async for _ in iter_agent_response_events(
                agent, "Read the index.", run_config=RunConfig(tracing_disabled=True)
            ):
                pass
        else:
            await Runner.run(agent, "Read the index.", run_config=RunConfig(tracing_disabled=True))
    except RequestCaptured:
        pass
    assert len(wire.requests) == 1
    request = wire.requests[0]
    body = json.loads(request.content)
    assert request.url.path.endswith("/responses")
    assert body["model"] == model
    assert body.get("stream") is True, body
    assert body["store"] is False
    assert body["tools"][0]["name"] == "read_file"


def sse_turn(text, tool=False):
    """Explicit protocol fixture, not a mocked LiteLLM/Agents final result."""
    message = {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    output = [message]
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_text.done",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
    ]
    if tool:
        call = {
            "id": "fc_fixture",
            "type": "function_call",
            "call_id": "call_fixture",
            "name": "read_file",
            "arguments": '{"path":"index.md"}',
            "status": "completed",
        }
        output.append(call)
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {**call, "arguments": "", "status": "in_progress"},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": call["id"],
                    "output_index": 1,
                    "delta": call["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": call["id"],
                    "output_index": 1,
                    "arguments": call["arguments"],
                },
                {"type": "response.output_item.done", "output_index": 1, "item": call},
            ]
        )
    response = {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6.2-sol",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    events.append({"type": "response.completed", "response": response})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': i})}\n\n"
        for i, event in enumerate(events)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_codex_sse_preserves_commentary_tool_roundtrip_and_final(wire, kb_dir, stream):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent, iter_agent_response_events

    wire.replies.extend([sse_turn("Checking wiki.", tool=True), sse_turn("Fixture answer.")])
    agent = build_query_agent(str(kb_dir / "wiki"), "chatgpt/gpt-6.2-sol")
    if stream:
        events = [
            event
            async for event in iter_agent_response_events(
                agent, "Read the index.", run_config=RunConfig(tracing_disabled=True)
            )
        ]
        assert any(e["event"] == "tool_call" and e["data"]["name"] == "read_file" for e in events)
        assert events[-1]["event"] == "final"
        assert "Checking wiki." in events[-1]["data"]["answer"]
        assert "Fixture answer." in events[-1]["data"]["answer"]
    else:
        result = await Runner.run(
            agent, "Read the index.", run_config=RunConfig(tracing_disabled=True)
        )
        assert result.final_output == "Fixture answer."
        assert any(getattr(item.raw_item, "name", None) == "read_file" for item in result.new_items)
        assert any(getattr(item.raw_item, "content", None) for item in result.new_items)
    assert len(wire.requests) == 2
    followup = json.loads(wire.requests[1].content)
    assert followup["stream"] is True
    assert any(item["type"] == "function_call_output" for item in followup["input"])


@pytest_asyncio.fixture
async def cached_wire(monkeypatch):
    """Real SDK cache and HTTP handlers; only auth and HTTP are substituted."""
    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.main import acompletion, completion

    state = SimpleNamespace(requests=[], clients=[])

    def capture(request):
        state.requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_turn("Offline lifecycle answer."),
        )

    def async_client(self, *args, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        state.clients.append(client)
        return client

    original_sync_init = HTTPHandler.__init__

    def sync_init(self, *args, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(capture))
        state.clients.append(client)
        kwargs["client"] = client
        original_sync_init(self, *args, **kwargs)

    def forbidden_auth(*args, **kwargs):
        raise AssertionError("OAuth file access forbidden")

    monkeypatch.setattr(Authenticator, "__init__", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_access_token", lambda self: "offline-token")
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: None)
    monkeypatch.setattr(Authenticator, "get_api_base", lambda self: "https://codex.invalid")
    monkeypatch.setattr(Authenticator, "_read_auth_file", forbidden_auth)
    monkeypatch.setattr(ChatGPTResponsesAPIConfig, "validate_environment", lambda *a, **k: {})
    monkeypatch.setattr(AsyncHTTPHandler, "create_client", async_client)
    monkeypatch.setattr(HTTPHandler, "__init__", sync_init)
    # A private test cache, never clear or close the application's global cache.
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "num_retries", 0)
    try:
        yield state
    finally:
        for client in state.clients:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
            else:
                client.close()


@pytest.mark.asyncio
async def test_compile_then_query_keeps_shared_sdk_client_open(cached_wire, kb_dir, monkeypatch):
    from agents import RunConfig

    from openkb.agent import compiler
    from openkb.agent.query import build_query_agent, iter_agent_response_events

    summary = kb_dir / "wiki" / "summaries" / "fixture.md"
    summary.write_text("# Fixture summary\n")
    model = "chatgpt/gpt-6.2-sol"

    async def compile_one_concept(*args, **kwargs):
        # Preserve real compiler sync overview + async concept calls and finally;
        # replace only the lengthy page-generation orchestration, not SDK calls.
        result = await compiler._llm_call_async(
            model, [{"role": "user", "content": "Offline concept."}], "concept"
        )
        assert result == "Offline lifecycle answer."

    monkeypatch.setattr(compiler, "_compile_concepts", compile_one_concept)
    await compiler.compile_long_doc("fixture", summary, "doc_fixture", kb_dir, model)
    assert len(cached_wire.requests) == 2
    async_clients = [c for c in cached_wire.clients if isinstance(c, httpx.AsyncClient)]
    assert len(async_clients) == 1
    # Do not stop at is_closed: exercise the exact Agents/SSE failure as well.
    agent = build_query_agent(str(kb_dir / "wiki"), model)
    events = [
        event
        async for event in iter_agent_response_events(
            agent, "Read the fixture.", run_config=RunConfig(tracing_disabled=True)
        )
    ]
    assert events[-1]["event"] == "final"
    assert events[-1]["data"]["answer"] == "Offline lifecycle answer."
    assert len(cached_wire.requests) == 3


@pytest.mark.asyncio
async def test_mixed_sync_async_calls_reuse_open_client_without_compiler_shutdown(
    cached_wire, kb_dir
):
    from agents import RunConfig

    from openkb.agent import compiler
    from openkb.agent.query import build_query_agent, iter_agent_response_events

    model = "chatgpt/gpt-6.2-sol"
    messages = [{"role": "user", "content": "Offline compiler control."}]
    assert compiler._llm_call(model, messages, "overview") == "Offline lifecycle answer."
    assert await compiler._llm_call_async(model, messages, "concept") == "Offline lifecycle answer."
    agent = build_query_agent(str(kb_dir / "wiki"), model)
    events = [
        event
        async for event in iter_agent_response_events(
            agent, "Read the fixture.", run_config=RunConfig(tracing_disabled=True)
        )
    ]
    assert events[-1]["event"] == "final"
    assert events[-1]["data"]["answer"] == "Offline lifecycle answer."
    assert len(cached_wire.requests) == 3
    async_clients = [c for c in cached_wire.clients if isinstance(c, httpx.AsyncClient)]
    assert len(async_clients) == 1
    assert not async_clients[0].is_closed


@pytest.mark.asyncio
async def test_upstream_shutdown_retains_closed_client_in_cache(cached_wire):
    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

    from openkb.agent import compiler

    model = "chatgpt/gpt-6.2-sol"
    messages = [{"role": "user", "content": "Offline lifecycle probe."}]
    assert await compiler._llm_call_async(model, messages, "concept") == "Offline lifecycle answer."
    handler = get_async_httpx_client(
        llm_provider=litellm.LlmProviders.CHATGPT, params={"ssl_verify": None}
    )
    assert not handler.client.is_closed
    await litellm.close_litellm_async_clients()
    assert handler.client.is_closed
    assert (
        get_async_httpx_client(
            llm_provider=litellm.LlmProviders.CHATGPT, params={"ssl_verify": None}
        )
        is handler
    )
    with pytest.raises(litellm.InternalServerError, match="client has been closed"):
        await litellm.acompletion(model=model, messages=messages, stream=True, num_retries=0)
    assert len(cached_wire.requests) == 1


def test_native_stream_override_is_provider_and_version_scoped(monkeypatch):
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    from openkb import provider_compat

    original = OpenAIResponsesAPIConfig.should_fake_stream
    monkeypatch.setattr(ChatGPTResponsesAPIConfig, "should_fake_stream", original)
    monkeypatch.setattr(provider_compat, "version", lambda name: "future-version")
    provider_compat.install_response_compat()
    assert ChatGPTResponsesAPIConfig.should_fake_stream is original
    monkeypatch.setattr(provider_compat, "version", lambda name: "1.87.2")
    provider_compat.install_response_compat()
    installed = ChatGPTResponsesAPIConfig.should_fake_stream
    provider_compat.install_response_compat()
    assert ChatGPTResponsesAPIConfig.should_fake_stream is installed
    assert OpenAIResponsesAPIConfig.should_fake_stream is original
    for stream in (True, False, None):
        assert installed(None, "gpt-6.2-sol", stream, "chatgpt") is False
