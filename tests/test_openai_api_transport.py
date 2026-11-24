"""API-key Responses route: real SDK serialization, no network/inference."""

import json

import pytest
import test_codex_stream_transport as transport

wire = transport.wire


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["compiler", "query", "stream"])
@pytest.mark.parametrize("model,effort", [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")])
async def test_openai_api_stage_wire(wire, kb_dir, stage, model, effort):
    from agents import RunConfig, Runner

    from openkb.agent import compiler
    from openkb.agent.query import build_query_agent
    from openkb.config import LlmCredentialBundle

    route = f"openai/responses/{model}"
    bundle = LlmCredentialBundle(api_key="offline-key", reasoning_effort=effort)
    agent = build_query_agent(str(kb_dir / "wiki"), route, bundle=bundle)
    if stage == "stream":
        assert agent.model_settings.include_usage is True
    try:
        if stage == "compiler":
            await compiler._llm_call_async(
                route, [{"role": "user", "content": "offline"}], "wire", bundle=bundle
            )
        elif stage == "query":
            await Runner.run(agent, "offline", run_config=RunConfig(tracing_disabled=True))
        else:
            result = Runner.run_streamed(
                agent, "offline", run_config=RunConfig(tracing_disabled=True)
            )
            async for _ in result.stream_events():
                pass
    except transport.RequestCaptured:
        pass
    assert len(wire.requests) == 1
    request = wire.requests[0]
    body = json.loads(request.content)
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/responses"
    assert body["model"] == model
    assert body["reasoning"]["effort"] == effort
    if stage == "stream":
        assert body.get("stream") is True
        assert body.get("tool_choice") != "none"
