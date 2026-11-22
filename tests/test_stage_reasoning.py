"""Offline reasoning configuration and request policy contracts."""

import pytest
from pydantic import ValidationError


def test_stage_reasoning_inheritance_and_validation():
    from openkb.stage_config import resolve_stage_reasoning

    assert resolve_stage_reasoning({}, "indexing") is None
    config = {
        "reasoning_effort": "medium",
        "indexing_reasoning_effort": "low",
        "query_reasoning_effort": None,
    }
    assert resolve_stage_reasoning(config, "indexing") == "low"
    assert resolve_stage_reasoning(config, "query") == "medium"
    for bad in ("", " LOW ", 1, False, "unknown"):
        with pytest.raises(ValueError):
            resolve_stage_reasoning({"query_reasoning_effort": bad}, "query")
    with pytest.raises(ValueError):
        resolve_stage_reasoning({}, "other")


def test_reasoning_config_schema_and_bundle_identity():
    from openkb.api_models import (
        GlobalConfigResponse,
        GlobalConfigValues,
        KbConfigResponse,
        _KbConfigWritable,
    )
    from openkb.config import GLOBAL_SCALAR_KEYS, LlmCredentialBundle
    from openkb.reasoning import with_reasoning_effort

    for key in (
        "reasoning_effort",
        "indexing_reasoning_effort",
        "compilation_reasoning_effort",
        "query_reasoning_effort",
    ):
        assert key in GLOBAL_SCALAR_KEYS
        for schema in (
            _KbConfigWritable,
            KbConfigResponse,
            GlobalConfigValues,
            GlobalConfigResponse,
        ):
            assert key in schema.model_fields
        with pytest.raises(ValidationError):
            _KbConfigWritable(**{key: "garbage"})
    bundle = LlmCredentialBundle(api_key="test-only")
    assert with_reasoning_effort(bundle, None) is bundle
    assert with_reasoning_effort(None, None) is None
    changed = with_reasoning_effort(bundle, "low")
    assert changed.reasoning_effort == "low"
    assert changed.api_key == bundle.api_key
    assert bundle.reasoning_effort is None


class UnsupportedEffort(Exception):
    status_code = 400


def test_single_leaf_fallback_without_quality_downgrade(caplog):
    from openkb.reasoning import call_with_reasoning_fallback

    calls = []
    response = object()

    def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnsupportedEffort(
                "Unsupported value: 'none' for parameter 'reasoning.effort'. secret-raw"
            )
        return response

    assert (
        call_with_reasoning_fallback(
            call, model="chatgpt/gpt-6-luna", messages=[], reasoning_effort="none"
        )
        is response
    )
    assert [x["reasoning_effort"] for x in calls] == ["none", "low"]
    assert "secret-raw" not in caplog.text
    assert "effective=low" in caplog.text


@pytest.mark.parametrize(
    "effort,stream,error",
    [
        ("medium", False, "Unsupported value reasoning.effort"),
        (None, False, "Unsupported value reasoning.effort"),
        ("none", True, "Unsupported value reasoning.effort"),
        ("none", False, "invalid request"),
        ("none", False, "reasoning.effort content policy"),
        ("none", False, "Unsupported reasoning.effort insufficient_quota"),
    ],
)
def test_no_unsafe_fallback(effort, stream, error):
    from openkb.reasoning import call_with_reasoning_fallback

    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        raise UnsupportedEffort(error)

    with pytest.raises(UnsupportedEffort):
        call_with_reasoning_fallback(
            call, model="chatgpt/gpt-6-sol", messages=[], reasoning_effort=effort, stream=stream
        )
    assert len(calls) == 1
    assert ("reasoning_effort" in calls[0]) == (effort is not None)


@pytest.mark.asyncio
async def test_async_fallback_stops_after_one_extra_request():
    from openkb.reasoning import acall_with_reasoning_fallback

    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        raise UnsupportedEffort("Unsupported value reasoning_effort")

    with pytest.raises(UnsupportedEffort):
        await acall_with_reasoning_fallback(
            call, model="chatgpt/gpt-6-luna", messages=[], reasoning_effort="none"
        )
    assert len(calls) == 2


def test_flash_reasoning_changes_cache_and_worker_request(tmp_path, monkeypatch):
    import sys

    from test_flash_adapter import result

    from openkb import flash_adapter

    calls = []

    def run(*args):
        calls.append(args[1])
        return result()

    monkeypatch.setattr(flash_adapter, "run_worker", run)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"offline")
    config = {"flash_python": sys.executable, "flash_cache_dir": str(tmp_path / "cache")}
    flash_adapter.index_pdf(pdf, config)
    flash_adapter.index_pdf(pdf, {**config, "indexing_reasoning_effort": "low"})
    assert len(calls) == 2
    assert calls[-1]["summary_reasoning_effort"] == "low"


def test_flash_worker_passes_summary_effort_to_sdk(monkeypatch):
    from unittest.mock import Mock

    from openkb import flash_worker

    monkeypatch.setattr(flash_worker, "version", lambda name: "0.7.4")
    client = Mock()
    client.submit_document.return_value = {"doc_id": "offline"}
    client.get_document.return_value = {"status": "completed", "name": "offline", "pageNum": 1}
    client.get_tree.return_value = {"status": "completed", "result": []}
    factory = Mock(return_value=client)
    flash_worker.execute(
        {
            "model": "chatgpt/gpt-6-luna",
            "pdf_path": "offline.pdf",
            "summary_reasoning_effort": "low",
        },
        client_factory=factory,
    )
    assert factory.call_args.kwargs["summary_reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_flash_reasoning_fallback_precedes_fatal_conversion(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pageindex

    from openkb.flash_worker import completion_policy

    utils = SimpleNamespace(
        _acompletion=None,
        _completion=None,
        _claude_none_kwargs=lambda m, k: k,
        _add_usage=lambda r: None,
    )
    monkeypatch.setitem(sys.modules, "pageindex.utils", utils)
    monkeypatch.setattr(pageindex, "utils", utils, raising=False)
    calls = []

    async def call(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise UnsupportedEffort("Unsupported value reasoning.effort")
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    import litellm

    monkeypatch.setattr(litellm, "acompletion", call)
    with completion_policy({"retries": 0}) as diagnostics:
        assert (
            await utils._acompletion("chatgpt/gpt-6-luna", [], {"reasoning_effort": "none"}) == "ok"
        )
    assert [c["reasoning_effort"] for c in calls] == ["none", "low"]
    assert diagnostics["calls"] == 2
    assert diagnostics["reasoning_fallbacks"] == 1
    assert diagnostics["requested_reasoning_effort"] == "none"
    assert diagnostics["effective_reasoning_effort"] == "low"


def test_config_reads_include_reasoning(monkeypatch, tmp_path):
    from openkb import api_config, config
    from openkb.config import LlmCredentialBundle

    values = {
        "model": "legacy",
        "language": "en",
        "pageindex_threshold": 20,
        "reasoning_effort": "high",
        "indexing_reasoning_effort": "low",
        "compilation_reasoning_effort": "medium",
        "query_reasoning_effort": "medium",
    }
    monkeypatch.setattr(api_config, "resolve_effective_config", lambda p: (values, {}))
    monkeypatch.setattr(api_config, "resolve_credential_bundle", lambda p: LlmCredentialBundle())
    monkeypatch.setattr(api_config, "load_global_config", lambda: values)
    monkeypatch.setattr(config, "GLOBAL_CONFIG_DIR", tmp_path)
    for response in (api_config.read_kb_config(tmp_path), api_config.read_global_config()):
        assert response.indexing_reasoning_effort == "low"
        assert response.compilation_reasoning_effort == "medium"
        assert response.query_reasoning_effort == "medium"
        assert response.reasoning_effort == "high"
    assert api_config.read_kb_config(tmp_path).global_values.query_reasoning_effort == "medium"


def test_query_agent_uses_bundle_reasoning(tmp_path):
    from openkb.agent.query import build_query_agent
    from openkb.config import LlmCredentialBundle

    agent = build_query_agent(
        str(tmp_path), "test", bundle=LlmCredentialBundle(reasoning_effort="medium")
    )
    assert agent.model_settings.reasoning.effort == "medium"


def test_chat_agent_resolves_query_effort_without_credentials(tmp_path, monkeypatch):
    from openkb import config
    from openkb.agent.query import build_chat_agent

    monkeypatch.setattr(
        config, "resolve_effective_config", lambda p: ({"query_reasoning_effort": "medium"}, {})
    )
    agent = build_chat_agent(tmp_path, "recorded-session-model")
    assert agent.model_settings.reasoning.effort == "medium"


def test_cli_chat_reasoning_preserves_legacy_headers(tmp_path, monkeypatch):
    from openkb import config
    from openkb.agent.query import build_chat_agent

    monkeypatch.setattr(config, "_runtime_extra_headers", {"X-Offline": "test"})
    monkeypatch.setattr(
        config, "resolve_effective_config", lambda p: ({"query_reasoning_effort": "medium"}, {})
    )
    agent = build_chat_agent(tmp_path, "recorded")
    assert agent.model_settings.extra_headers == {"X-Offline": "test"}


def test_cache_preserves_only_validated_reasoning_diagnostics():
    from test_flash_adapter import result

    from openkb.flash_adapter import validate_result

    diagnostics = {
        "calls": 2,
        "retries": 1,
        "reasoning_fallbacks": 1,
        "requested_reasoning_effort": "none",
        "effective_reasoning_effort": "low",
        "raw_provider_error": "secret",
    }
    clean = validate_result({**result(), "diagnostics": diagnostics})["diagnostics"]
    assert clean["reasoning_fallbacks"] == 1
    assert clean["requested_reasoning_effort"] == "none"
    assert clean["effective_reasoning_effort"] == "low"
    assert "raw_provider_error" not in clean
    with pytest.raises(ValueError):
        validate_result(
            {**result(), "diagnostics": {**diagnostics, "effective_reasoning_effort": "raw-secret"}}
        )


@pytest.mark.asyncio
async def test_rest_stream_query_applies_query_effort(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openkb import api_helpers
    from openkb.config import LlmCredentialBundle

    seen = []
    monkeypatch.setattr(
        api_helpers,
        "resolve_effective_config",
        lambda p: ({"query_reasoning_effort": "medium"}, {}),
    )
    monkeypatch.setattr(api_helpers, "build_run_config_from_bundle", lambda *a: None)

    def build(*args, **kwargs):
        seen.append(kwargs["bundle"])
        return object()

    async def events(*a, **kw):
        yield {"event": "final", "data": {"answer": "offline", "history": []}}

    async def disconnected():
        return False

    monkeypatch.setattr(api_helpers, "build_query_agent", build)
    monkeypatch.setattr(api_helpers, "iter_agent_response_events", events)
    monkeypatch.setattr(api_helpers, "append_log", lambda *a: None)
    request = SimpleNamespace(question="offline", save=False)
    frames = [
        frame
        async for frame in api_helpers._stream_query(
            request,
            tmp_path,
            "chatgpt/gpt-6-sol",
            SimpleNamespace(is_disconnected=disconnected),
            bundle=LlmCredentialBundle(),
        )
    ]
    assert seen[0].reasoning_effort == "medium"
    assert any("final" in frame for frame in frames)


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_status_errors_never_trigger_reasoning_retry(status):
    from openkb.reasoning import call_with_reasoning_fallback

    error = UnsupportedEffort("Unsupported reasoning.effort")
    error.status_code = status
    calls = []

    def call(**kw):
        calls.append(kw)
        raise error

    with pytest.raises(UnsupportedEffort):
        call_with_reasoning_fallback(
            call, model="chatgpt/gpt-6-sol", messages=[], reasoning_effort="none"
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_stream_error_after_visible_tool_event_is_not_replayed():
    from openkb.reasoning import acall_with_reasoning_fallback

    calls = []

    async def stream():
        yield {"event": "tool_call", "data": {"name": "read_file"}}
        raise UnsupportedEffort("Unsupported reasoning.effort")

    async def call(**kw):
        calls.append(kw)
        return stream()

    response = await acall_with_reasoning_fallback(
        call, model="chatgpt/gpt-6-sol", messages=[], reasoning_effort="none", stream=True
    )
    events = []
    with pytest.raises(UnsupportedEffort):
        async for event in response:
            events.append(event)
    assert len(calls) == 1
    assert events[0]["event"] == "tool_call"


@pytest.mark.asyncio
async def test_bounded_fallback_remains_single_across_transient_retry():
    from openkb.indexing_policy import FlashAbort, bounded_completion

    calls = []

    async def call(**kw):
        calls.append(kw)
        error = UnsupportedEffort("Unsupported reasoning.effort")
        if len(calls) == 2:
            error.status_code = 500
        raise error

    async def sleep(delay):
        pass

    with pytest.raises(FlashAbort):
        await bounded_completion(
            call, "chatgpt/gpt-6-luna", [], {"reasoning_effort": "none"}, retries=1, sleep=sleep
        )
    assert [c["reasoning_effort"] for c in calls] == ["none", "low", "low"]


def test_wrapped_quota_cannot_trigger_reasoning_fallback():
    from openkb.reasoning import call_with_reasoning_fallback

    outer = UnsupportedEffort("Unsupported reasoning.effort")
    outer.__cause__ = RuntimeError("usage_limit_reached")
    calls = []

    def call(**kw):
        calls.append(kw)
        raise outer

    with pytest.raises(UnsupportedEffort):
        call_with_reasoning_fallback(
            call, model="chatgpt/gpt-6-sol", messages=[], reasoning_effort="none"
        )
    assert len(calls) == 1


def test_success_logs_requested_effective_without_raw_response(caplog):
    import logging

    from openkb.reasoning import call_with_reasoning_fallback

    with caplog.at_level(logging.INFO, logger="openkb.reasoning"):
        response = call_with_reasoning_fallback(
            lambda **kw: "secret-response",
            model="chatgpt/gpt-6-sol",
            messages=[],
            reasoning_effort="medium",
        )
    assert response == "secret-response"
    assert "requested=medium effective=medium fallback=false" in caplog.text
    assert "secret-response" not in caplog.text
