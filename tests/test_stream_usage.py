"""Pinned SDK streaming must expose provider-reported usage (offline SSE only)."""

import pytest
import test_codex_stream_transport as transport

wire = transport.wire


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["chatgpt", "openai/responses"])
@pytest.mark.parametrize("with_bundle", [False, True])
@pytest.mark.parametrize("tool_turn", [False, True])
async def test_chatgpt_stream_exposes_usage_for_every_completed_model_turn(
    wire, kb_dir, monkeypatch, with_bundle, tool_turn, provider
):
    import litellm
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent, build_run_config_from_bundle
    from openkb.config import LlmCredentialBundle

    if tool_turn:
        wire.replies.append(
            transport.sse_turn("Read the index.", tool=True).replace("gpt-6.2-sol", "gpt-6-sol")
        )
    wire.replies.append(transport.sse_turn("Offline answer.").replace("gpt-6.2-sol", "gpt-6-sol"))
    original = litellm.acompletion
    seen = []
    settings = []

    async def observe(*args, **kwargs):
        settings.append(kwargs.get("stream_options"))
        response = await original(*args, **kwargs)

        async def chunks():
            async for chunk in response:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    seen.append(usage)
                yield chunk

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", observe)
    bundle = LlmCredentialBundle(reasoning_effort="medium") if with_bundle else None
    agent = build_query_agent(
        str(kb_dir / "wiki"),
        f"{provider}/gpt-6-sol",
        bundle=bundle,
    )
    config = build_run_config_from_bundle(f"{provider}/gpt-6-sol", bundle) or RunConfig()
    config.tracing_disabled = True
    result = Runner.run_streamed(agent, "Offline usage fixture", run_config=config)
    async for _ in result.stream_events():
        pass
    assert seen, "Provider usage was lost before the metering boundary"
    assert seen[-1].total_tokens == 15
    turns = 2 if tool_turn else 1
    assert len(seen) == turns
    assert settings == [{"include_usage": True}] * turns
    assert len(wire.requests) == turns
    assert result.raw_responses[-1].usage.total_tokens == 15


def test_other_providers_do_not_gain_stream_options(kb_dir):
    from openkb.agent.query import build_query_agent

    agent = build_query_agent(str(kb_dir / "wiki"), "openai/offline-fixture")
    assert agent.model_settings.include_usage is None
