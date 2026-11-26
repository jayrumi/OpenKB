"""Mock-only end-to-end checks of stage selection at CLI/REST boundaries."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from openkb.cli import cli

CONFIG = {
    "model": "legacy/exact-fallback",
    "indexing_model": "chatgpt/gpt-6-luna",
    "compilation_model": "chatgpt/gpt-6.2-sol",
    "query_model": "query/exact-token",
    "language": "en",
}


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    import openkb.config as config

    monkeypatch.setattr(config, "GLOBAL_CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp_path / "global" / "global.yaml")
    monkeypatch.setenv("OPENKB_KB_ROOT", str(tmp_path / "kbs"))
    monkeypatch.delenv("OPENKB_API_TOKEN", raising=False)
    for module in ("openkb.cli", "openkb.api", "openkb.api_helpers", "openkb.config"):
        monkeypatch.setattr(
            importlib.import_module(module),
            "resolve_effective_config",
            lambda *_: (dict(CONFIG), {}),
        )
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *_: None)
    monkeypatch.setattr("openkb.api_helpers.resolve_kb_alias", lambda *_: tmp_path)


def test_cli_query_uses_query_model(kb_dir):
    with (
        patch("openkb.cli._find_kb_dir", return_value=kb_dir),
        patch("openkb.agent.query.run_query", new_callable=AsyncMock, return_value="answer") as run,
    ):
        result = CliRunner().invoke(cli, ["query", "question"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[2] == CONFIG["query_model"]


@pytest.mark.parametrize("long_doc", [False, True])
def test_add_passes_compilation_model_not_indexing_model(kb_dir, long_doc):
    from openkb.cli import add_single_file
    from openkb.converter import ConvertResult

    source = kb_dir / "wiki" / "sources" / "notes.md"
    source.write_text("# Notes")
    raw = kb_dir / "raw" / "notes.md"
    raw.write_text("# Notes")
    converted = ConvertResult(
        raw_path=raw, source_path=source, doc_name="notes", is_long_doc=long_doc
    )
    with (
        patch("openkb.cli.convert_document", return_value=converted),
        patch(
            "openkb.indexer.index_long_document",
            return_value=SimpleNamespace(doc_id="id", description="desc"),
        ),
        patch("openkb.agent.compiler.compile_short_doc", new_callable=AsyncMock) as short,
        patch("openkb.agent.compiler.compile_long_doc", new_callable=AsyncMock) as long,
    ):
        assert add_single_file(raw, kb_dir) == "added"
    call = long.call_args if long_doc else short.call_args
    assert call.args[4 if long_doc else 3] == CONFIG["compilation_model"]


def test_cli_chat_new_session_uses_query_model(kb_dir):
    with (
        patch("openkb.cli._find_kb_dir", return_value=kb_dir),
        patch("openkb.agent.chat.run_chat", new_callable=AsyncMock) as run,
    ):
        result = CliRunner().invoke(cli, ["chat"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[1].model == CONFIG["query_model"]


@pytest.mark.parametrize("target", ["deck", "skill"])
def test_cli_generation_uses_compilation_model(kb_dir, target):
    (kb_dir / "wiki" / "concepts" / "foo.md").write_text("# Foo")
    with (
        patch("openkb.cli._find_kb_dir", return_value=kb_dir),
        patch("openkb.skill.generator.Generator") as factory,
    ):
        factory.return_value.run = AsyncMock()
        factory.return_value.validation = SimpleNamespace(errors=[], warnings=[], ok=True)
        result = CliRunner().invoke(cli, [target, "new", "sample", "Explain this"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert factory.call_args.kwargs["model"] == CONFIG["compilation_model"]


@pytest.mark.parametrize("stream", [False, True])
def test_rest_query_uses_request_local_query_model_for_agent_and_run_config(kb_dir, stream):
    from openkb.api import create_app

    async def events(*args, **kwargs):
        yield {"event": "final", "data": {"answer": "answer", "history": []}}

    with (
        patch("openkb.api.run_query", new_callable=AsyncMock, return_value="answer") as run,
        patch("openkb.api.build_run_config_from_bundle", return_value=None) as config,
        patch(
            "openkb.api_helpers.build_run_config_from_bundle", return_value=None
        ) as stream_config,
        patch("openkb.api_helpers.build_query_agent", return_value=object()) as agent,
        patch("openkb.api_helpers.iter_agent_response_events", events),
    ):
        response = TestClient(create_app()).post(
            "/api/v1/query", json={"kb": "test", "question": "What?", "stream": stream}
        )
    assert response.status_code == 200, response.text
    assert config.call_args.args[0] == CONFIG["query_model"]
    if stream:
        assert agent.call_args.args[1] == CONFIG["query_model"]
        assert stream_config.call_args.args[0] == CONFIG["query_model"]
    else:
        assert run.call_args.args[2] == CONFIG["query_model"]


def test_rest_new_chat_session_uses_query_model_and_resume_keeps_recorded_model(kb_dir):
    from openkb.api_helpers import _load_or_create_session

    session = _load_or_create_session(kb_dir, None)
    assert session.model == CONFIG["query_model"]
    session.model = "historical/session-model"
    session.save()
    assert _load_or_create_session(kb_dir, session.id).model == "historical/session-model"


@pytest.mark.parametrize("target", ["deck", "skill"])
@pytest.mark.asyncio
async def test_rest_generation_uses_compilation_model(kb_dir, target):
    from openkb import api_helpers
    from openkb.api_models import DeckRequest, SkillRequest

    request_class = DeckRequest if target == "deck" else SkillRequest
    request = request_class(kb="test", name="sample", intent="Explain")
    with (
        patch("openkb.cli._preflight_skill_new", return_value=None),
        patch("openkb.skill.generator.Generator") as factory,
    ):
        factory.return_value.run = AsyncMock()
        factory.return_value.output_dir = kb_dir / "output" / target
        events = [event async for event in getattr(api_helpers, f"_iter_{target}")(request, kb_dir)]
    assert events[-1]["event"] == "final", events
    assert factory.call_args.kwargs["model"] == CONFIG["compilation_model"]


@pytest.mark.asyncio
async def test_rest_semantic_lint_uses_compilation_model(kb_dir):
    from openkb.cli import run_lint_report

    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({"hash": {"name": "notes"}}))
    with (
        patch("openkb.lint.run_structural_lint", return_value="ok"),
        patch(
            "openkb.agent.linter.run_knowledge_lint", new_callable=AsyncMock, return_value="ok"
        ) as run,
    ):
        await run_lint_report(kb_dir)
    assert run.call_args.args[1] == CONFIG["compilation_model"]


@pytest.mark.parametrize("target", ["deck", "skill"])
@pytest.mark.asyncio
async def test_chat_slash_generation_uses_compilation_model(kb_dir, target):
    from prompt_toolkit.styles import Style

    from openkb.agent.chat import _handle_slash
    from openkb.agent.chat_session import ChatSession

    (kb_dir / "wiki" / "concepts" / "foo.md").write_text("# Foo")
    session = ChatSession.new(kb_dir, CONFIG["query_model"], "en")
    with patch("openkb.skill.generator.Generator") as factory:
        factory.return_value.run = AsyncMock()
        factory.return_value.validation = SimpleNamespace(errors=[], warnings=[], ok=True)
        await _handle_slash(f'/{target} new sample "Explain"', kb_dir, session, Style.from_dict({}))
    assert factory.call_args.kwargs["model"] == CONFIG["compilation_model"]
    assert session.model == CONFIG["query_model"]


@pytest.mark.asyncio
async def test_chat_critique_uses_compilation_model(kb_dir):
    from prompt_toolkit.styles import Style

    from openkb.agent.chat import _handle_slash
    from openkb.agent.chat_session import ChatSession

    (kb_dir / "output").mkdir()
    (kb_dir / "output" / "demo.html").write_text("<html></html>")
    session = ChatSession.new(kb_dir, CONFIG["query_model"], "en")
    with patch("openkb.agent.skill_runner.run_skill", new_callable=AsyncMock) as run:
        await _handle_slash("/critique output/demo.html", kb_dir, session, Style.from_dict({}))
    assert run.call_args.kwargs["model"] == CONFIG["compilation_model"]


def test_pageindex_cleanup_uses_indexing_model(kb_dir, monkeypatch):
    from openkb.cli import _cleanup_pageindex

    monkeypatch.setitem(CONFIG, "indexing_model", "indexing/exact-token")

    (kb_dir / ".openkb" / "pageindex.db").touch()
    with patch("pageindex.PageIndexClient") as factory:
        assert _cleanup_pageindex(kb_dir / ".openkb", kb_dir, "notes", "id")[0]
    assert factory.call_args.kwargs["model"] == CONFIG["indexing_model"]


def test_pageindex_cleanup_chatgpt_uses_oauth_aware_client(kb_dir):
    from openkb.cli import _cleanup_pageindex

    (kb_dir / ".openkb" / "pageindex.db").touch()
    with patch("openkb.provider_compat.OAuthPageIndexClient") as factory:
        assert _cleanup_pageindex(kb_dir / ".openkb", kb_dir, "notes", "id")[0]
    assert factory.call_args.kwargs["model"] == "chatgpt/gpt-6-luna"


def test_skill_eval_uses_compilation_model(kb_dir):
    target = kb_dir / "output" / "skills" / "sample"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("---\nname: sample\ndescription: Sample\n---\n# Sample")
    with (
        patch("openkb.cli._find_kb_dir", return_value=kb_dir),
        patch(
            "openkb.skill.evaluator.run_eval",
            new_callable=AsyncMock,
            side_effect=RuntimeError("mock stop"),
        ) as run,
    ):
        result = CliRunner().invoke(cli, ["skill", "eval", "sample"])
    assert "mock stop" in result.output
    assert run.call_args.kwargs["model"] == CONFIG["compilation_model"]
