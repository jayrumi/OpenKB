"""Ancillary compilation paths keep their request-local stage effort."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import test_stage_model_wiring as wiring
from click.testing import CliRunner

CONFIG = wiring.CONFIG
isolated_config = wiring.isolated_config


@pytest.fixture(autouse=True)
def reasoning_config(isolated_config, monkeypatch):
    monkeypatch.setitem(CONFIG, "compilation_model", "chatgpt/gpt-6-sol")
    monkeypatch.setitem(CONFIG, "compilation_reasoning_effort", "medium")
    monkeypatch.setitem(CONFIG, "query_reasoning_effort", "low")


@pytest.mark.parametrize("target", ["deck", "skill"])
def test_cli_generation_passes_compilation_effort(kb_dir, target):
    from openkb.cli import cli

    (kb_dir / "wiki" / "concepts" / "foo.md").write_text("# Foo")
    with (
        patch("openkb.cli._find_kb_dir", return_value=kb_dir),
        patch("openkb.skill.generator.Generator") as factory,
    ):
        factory.return_value.run = AsyncMock()
        factory.return_value.validation = SimpleNamespace(errors=[], warnings=[], ok=True)
        result = CliRunner().invoke(cli, [target, "new", "sample", "Explain this"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert factory.call_args.kwargs["bundle"].reasoning_effort == "medium"


@pytest.mark.parametrize("target", ["deck", "skill"])
@pytest.mark.asyncio
async def test_rest_generation_passes_compilation_effort(kb_dir, target):
    from openkb import api_helpers
    from openkb.api_models import DeckRequest, SkillRequest

    request = (DeckRequest if target == "deck" else SkillRequest)(
        kb="test", name="sample", intent="Explain"
    )
    with (
        patch("openkb.cli._preflight_skill_new", return_value=None),
        patch("openkb.skill.generator.Generator") as factory,
    ):
        factory.return_value.run = AsyncMock()
        factory.return_value.output_dir = kb_dir / "output" / target
        events = [e async for e in getattr(api_helpers, f"_iter_{target}")(request, kb_dir)]
    assert events[-1]["event"] == "final", events
    assert factory.call_args.kwargs["bundle"].reasoning_effort == "medium"


@pytest.mark.parametrize("target", ["deck", "skill"])
@pytest.mark.asyncio
async def test_chat_slash_generation_passes_compilation_effort(kb_dir, target):
    from prompt_toolkit.styles import Style

    from openkb.agent.chat import _handle_slash
    from openkb.agent.chat_session import ChatSession

    (kb_dir / "wiki" / "concepts" / "foo.md").write_text("# Foo")
    session = ChatSession.new(kb_dir, "recorded", "en")
    with patch("openkb.skill.generator.Generator") as factory:
        factory.return_value.run = AsyncMock()
        factory.return_value.validation = SimpleNamespace(errors=[], warnings=[], ok=True)
        await _handle_slash(f'/{target} new sample "Explain"', kb_dir, session, Style.from_dict({}))
    assert factory.call_args.kwargs["bundle"].reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_semantic_lint_passes_stage_effort_to_run_config(kb_dir):
    import json

    from openkb.cli import run_lint, run_lint_report

    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({"hash": {"name": "notes"}}))
    for lint in (run_lint_report, run_lint):
        with (
            patch("openkb.lint.run_structural_lint", return_value="ok"),
            patch(
                "openkb.agent.linter.run_knowledge_lint", new_callable=AsyncMock, return_value="ok"
            ) as run,
        ):
            await lint(kb_dir)
        assert run.call_args.kwargs["run_config"].model_settings.reasoning.effort == "medium"


@pytest.mark.asyncio
async def test_chat_critique_passes_compilation_effort(kb_dir):
    from prompt_toolkit.styles import Style

    from openkb.agent.chat import _handle_slash
    from openkb.agent.chat_session import ChatSession

    (kb_dir / "output").mkdir(exist_ok=True)
    (kb_dir / "output" / "demo.html").write_text("<html></html>")
    session = ChatSession.new(kb_dir, "recorded", "en")
    with patch("openkb.agent.skill_runner.run_skill", new_callable=AsyncMock) as run:
        await _handle_slash("/critique output/demo.html", kb_dir, session, Style.from_dict({}))
    assert run.call_args.kwargs["bundle"].reasoning_effort == "medium"


def test_run_config_has_explicit_reasoning():
    from openkb.agent.query import build_run_config_from_bundle
    from openkb.config import LlmCredentialBundle

    config = build_run_config_from_bundle(
        "chatgpt/gpt-6-sol", LlmCredentialBundle(reasoning_effort="medium")
    )
    assert config.model_settings.reasoning.effort == "medium"
    assert config.model_settings.extra_args["allowed_openai_params"] == ["reasoning_effort"]
