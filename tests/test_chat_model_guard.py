"""Deployment chat allowlist rejects resumes without rewriting stored history."""

import json

import pytest
from fastapi.testclient import TestClient

from openkb.agent.chat_session import ChatSession, chats_dir
from openkb.api import create_app

LEGACY = "chatgpt/responses/gpt-6.1-sol"
APPROVED = "openai/responses/gpt-6-sol"


def forbidden(*args, **kwargs):
    raise AssertionError("Refused chat must not resolve credentials, build agents, or run tools")


@pytest.fixture
def saved_session(kb_dir):
    session = ChatSession.new(kb_dir, LEGACY, "en")
    session.record_turn(
        "Original question",
        "Original answer",
        [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Original answer"},
        ],
    )
    return session


@pytest.mark.parametrize("stream", [False, True])
def test_api_refuses_legacy_resume_before_credentials_and_stream_work(
    monkeypatch, kb_dir, saved_session, stream
):
    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([APPROVED]))
    monkeypatch.delenv("OPENKB_API_TOKEN", raising=False)
    monkeypatch.setattr("openkb.api._resolve_kb", lambda kb: kb_dir)
    monkeypatch.setattr("openkb.api.resolve_credential_bundle", forbidden)
    monkeypatch.setattr("openkb.api.build_run_config_from_bundle", forbidden)
    monkeypatch.setattr("openkb.api.build_chat_session_agent", forbidden)
    monkeypatch.setattr("openkb.api_helpers.build_chat_session_agent", forbidden)
    path = chats_dir(kb_dir) / f"{saved_session.id}.json"
    before = path.read_bytes()
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "kb": "test-kb",
                "session_id": saved_session.id,
                "message": "Continue",
                "stream": stream,
            },
        )
    assert response.status_code == 409
    assert "start a new chat" in response.json()["detail"].lower()
    assert LEGACY in response.json()["detail"]
    assert path.read_bytes() == before
    assert not (kb_dir / "wiki" / "log.md").exists()


def test_cli_refuses_resume_before_key_setup(monkeypatch, kb_dir, saved_session):
    from click.testing import CliRunner

    from openkb.cli import cli

    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([APPROVED]))
    monkeypatch.setattr("openkb.cli._find_kb_dir", lambda override: kb_dir)
    monkeypatch.setattr("openkb.cli.resolve_effective_config", lambda kb: ({}, {}))
    monkeypatch.setattr("openkb.cli._setup_llm_key", forbidden)
    monkeypatch.setattr("openkb.agent.chat.run_chat", forbidden)
    path = chats_dir(kb_dir) / f"{saved_session.id}.json"
    before = path.read_bytes()
    result = CliRunner().invoke(cli, ["chat", "--resume", saved_session.id])
    assert result.exit_code == 1
    assert "start a new chat" in result.output.lower()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-json",
        "null",
        "true",
        "42",
        '"legacy"',
        '{"legacy": true}',
        "[null]",
        "[42]",
        '[""]',
        '["  "]',
    ],
)
def test_invalid_deployment_allowlist_fails_closed(monkeypatch, raw):
    from openkb.stage_config import require_chat_model_allowed

    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", raw)
    with pytest.raises(ValueError, match="OPENKB_CHAT_ALLOWED_MODELS must be a JSON list"):
        require_chat_model_allowed("legacy")


def test_allowlist_is_exact_opt_in_and_not_cached(monkeypatch):
    from openkb.stage_config import require_chat_model_allowed

    monkeypatch.delenv("OPENKB_CHAT_ALLOWED_MODELS", raising=False)
    require_chat_model_allowed(LEGACY)
    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([LEGACY, APPROVED]))
    require_chat_model_allowed(LEGACY)
    require_chat_model_allowed(APPROVED)
    with pytest.raises(ValueError, match="start a new chat"):
        require_chat_model_allowed(" " + APPROVED)
    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", "[]")
    with pytest.raises(ValueError, match="start a new chat"):
        require_chat_model_allowed(APPROVED)


@pytest.mark.parametrize("allowed", [None, [LEGACY], [LEGACY, APPROVED]])
def test_compatible_resume_preserves_recorded_model(monkeypatch, kb_dir, saved_session, allowed):
    from openkb.api_helpers import _load_or_create_session

    if allowed is None:
        monkeypatch.delenv("OPENKB_CHAT_ALLOWED_MODELS", raising=False)
    else:
        monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps(allowed))
    monkeypatch.setattr("openkb.api_helpers.resolve_effective_config", forbidden)
    resumed = _load_or_create_session(kb_dir, saved_session.id)
    assert resumed.model == LEGACY
    assert resumed.history == saved_session.history


def test_new_chat_uses_query_stage_model_without_rewriting_history(
    monkeypatch, kb_dir, saved_session
):
    from openkb.api_helpers import _load_or_create_session

    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([APPROVED]))
    monkeypatch.setattr(
        "openkb.api_helpers.resolve_effective_config",
        lambda kb: ({"model": LEGACY, "query_model": APPROVED}, {}),
    )
    path = chats_dir(kb_dir) / f"{saved_session.id}.json"
    before = path.read_bytes()
    new = _load_or_create_session(kb_dir, None)
    assert new.model == APPROVED
    assert new.id != saved_session.id
    assert new.history == []
    assert path.read_bytes() == before


def test_saved_legacy_history_remains_viewable(monkeypatch, kb_dir, saved_session):
    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([APPROVED]))
    monkeypatch.delenv("OPENKB_API_TOKEN", raising=False)
    monkeypatch.setattr("openkb.api._resolve_kb", lambda kb: kb_dir)
    monkeypatch.setattr("openkb.api.resolve_credential_bundle", forbidden)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/chat/sessions/load",
        json={
            "kb": "test-kb",
            "session_id": saved_session.id,
        },
    )
    assert response.status_code == 200
    assert response.json()["user_turns"] == saved_session.user_turns
    assert response.json()["assistant_texts"] == saved_session.assistant_texts


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", [LEGACY, APPROVED])
def test_api_allowed_resume_reaches_turn_with_stored_model(monkeypatch, kb_dir, stream, model):
    session = ChatSession.new(kb_dir, model, "en")
    session.record_turn("Earlier", "Answer", [{"role": "user", "content": "Earlier"}])
    monkeypatch.setenv("OPENKB_CHAT_ALLOWED_MODELS", json.dumps([model]))
    monkeypatch.delenv("OPENKB_API_TOKEN", raising=False)
    monkeypatch.setattr("openkb.api._resolve_kb", lambda kb: kb_dir)
    monkeypatch.setattr("openkb.api.resolve_credential_bundle", lambda kb: None)

    def build_agent(kb, resumed, **kwargs):
        assert resumed.model == model
        assert resumed.history == session.history
        return object()

    async def events(agent, resumed, message, **kwargs):
        assert resumed.model == model
        yield {"event": "final", "data": {"answer": "Offline answer"}}

    for module in ("openkb.api", "openkb.api_helpers"):
        monkeypatch.setattr(f"{module}.build_run_config_from_bundle", lambda *args: None)
        monkeypatch.setattr(f"{module}.build_chat_session_agent", build_agent)
        monkeypatch.setattr(f"{module}.iter_chat_turn_events", events)
    response = TestClient(create_app()).post(
        "/api/v1/chat",
        json={
            "kb": "test-kb",
            "session_id": session.id,
            "message": "Continue",
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert "Offline answer" in response.text
