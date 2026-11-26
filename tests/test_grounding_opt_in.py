"""Ordinary compilation publishes without an implicit factual-audit gate."""

import json

import pytest

from openkb.agent import compiler


@pytest.mark.parametrize("value", ["false", "true", 0, 1, [], {}])
def test_grounding_audit_rejects_non_boolean_config(value):
    from openkb.config import resolve_grounding_audit

    with pytest.raises(ValueError, match="grounding_audit.*boolean"):
        resolve_grounding_audit({"grounding_audit": value})


def test_grounding_audit_config_defaults_and_layers(tmp_path, monkeypatch):
    from openkb import config

    monkeypatch.setattr(config, "load_global_config", lambda: {"grounding_audit": True})
    assert config.DEFAULT_CONFIG["grounding_audit"] is False
    assert config.resolve_grounding_audit({}) is False
    assert config.resolve_grounding_audit(config.resolve_effective_config(tmp_path)[0]) is True
    (tmp_path / ".openkb").mkdir()
    (tmp_path / ".openkb/config.yaml").write_text("grounding_audit: false\n")
    effective, sources = config.resolve_effective_config(tmp_path)
    assert config.resolve_grounding_audit(effective) is False
    assert sources["grounding_audit"] == "kb"


@pytest.mark.parametrize("global_value", [False, True])
def test_grounding_audit_kb_empty_list_is_not_inheritance(tmp_path, monkeypatch, global_value):
    from openkb import config

    monkeypatch.setattr(config, "load_global_config", lambda: {"grounding_audit": global_value})
    (tmp_path / ".openkb").mkdir()
    (tmp_path / ".openkb/config.yaml").write_text("grounding_audit: []\n")
    effective, sources = config.resolve_effective_config(tmp_path)
    with pytest.raises(ValueError, match="grounding_audit.*boolean"):
        config.resolve_grounding_audit(effective)
    assert sources["grounding_audit"] == "kb"


def test_grounding_audit_api_is_strict_boolean():
    from pydantic import ValidationError

    from openkb.api_models import _KbConfigWritable

    assert _KbConfigWritable(grounding_audit=True).model_dump()["grounding_audit"] is True
    for value in ("false", "true", 0, 1):
        with pytest.raises(ValidationError):
            _KbConfigWritable(grounding_audit=value)


def test_grounding_audit_global_null_reads_as_default(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openkb import api_config, config

    monkeypatch.setattr(config, "GLOBAL_CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp_path / "global/global.yaml")
    config.save_global_config({"grounding_audit": None})
    monkeypatch.setattr(
        api_config,
        "resolve_credential_bundle",
        lambda kb: SimpleNamespace(base_url=None, api_key=None),
    )
    assert config.resolve_grounding_audit(config.resolve_effective_config(tmp_path)[0]) is False
    assert api_config.read_global_config().grounding_audit is False
    response = api_config.read_kb_config(tmp_path)
    assert response.grounding_audit is False
    assert response.global_values.grounding_audit is None  # Raw unset layer, not effective value.
    assert response.sources["grounding_audit"] == "default"


def test_grounding_audit_api_round_trip_and_inheritance(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openkb import api_config, config
    from openkb.api_models import GlobalConfigPatchRequest, KbConfigPatchRequest

    monkeypatch.setattr(config, "GLOBAL_CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp_path / "global/global.yaml")
    monkeypatch.setattr(
        api_config,
        "resolve_credential_bundle",
        lambda kb: SimpleNamespace(base_url=None, api_key=None),
    )
    assert api_config.read_global_config().grounding_audit is False
    api_config.apply_global_config_patch(GlobalConfigPatchRequest(config={"grounding_audit": True}))
    assert api_config.read_global_config().grounding_audit is True
    inherited = api_config.read_kb_config(tmp_path)
    assert inherited.grounding_audit is True
    assert inherited.global_values.grounding_audit is True
    assert inherited.sources["grounding_audit"] == "global"
    api_config.apply_kb_config_patch(
        tmp_path, KbConfigPatchRequest(kb="test", config={"grounding_audit": False})
    )
    assert api_config.read_kb_config(tmp_path).grounding_audit is False
    api_config.apply_kb_config_patch(
        tmp_path, KbConfigPatchRequest(kb="test", config={"grounding_audit": None})
    )
    assert api_config.read_kb_config(tmp_path).grounding_audit is True


@pytest.mark.asyncio
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("source_data", [None, "[]", '[{"page": 1, "content": "  "}]', "{}"])
async def test_long_doc_requires_original_source_without_audit(
    tmp_path, monkeypatch, planned, source_data
):
    from openkb.grounding import GroundingError

    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    if source_data is not None:
        (wiki / "sources/report.json").write_text(source_data)
    summary = wiki / "summaries/report.md"
    original = "Mosaic provides code review, code search, and IAM."
    summary.write_text(original)
    monkeypatch.setattr("openkb.config.resolve_effective_config", lambda kb: ({}, {}))

    def sync(model, messages, step, **kw):
        if step == "concepts-plan":
            return json.dumps({"create": [{"name": "review"}]} if planned else {})
        return original

    async def generate(*args, **kwargs):
        return json.dumps({"content": original})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", generate)
    with pytest.raises(GroundingError, match="source"):
        await compiler.compile_long_doc("report", summary, "id", tmp_path, "mock/model")
    assert summary.read_text() == original
    assert not (wiki / "index.md").exists()
    assert not (wiki / "concepts/review.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", ["concepts", "entities"])
async def test_unaudited_update_retains_historical_source_context(tmp_path, monkeypatch, folder):
    wiki = tmp_path / "wiki"
    for directory in ("sources", "summaries", folder):
        (wiki / directory).mkdir(parents=True)
    historical = "Mosaic was founded in 2001. [source doc-a p. 2]"
    old_footer = "Source attribution evidence: [source doc-a p. 2]"
    prior = (
        "---\nsources: [summaries/doc-a.md]\ntype: product\n"
        "description: Mosaic history\n---\n" + historical + "\n\n" + old_footer + "\n"
    )
    page = wiki / folder / "mosaic.md"
    page.write_text(prior)
    current = "Mosaic provides review, search, and annotations."
    source_path = wiki / "sources/doc-b.json"
    source_path.write_text(json.dumps([{"page": 1, "content": current}]))
    summary = wiki / "summaries/doc-b.md"
    summary.write_text(current)
    monkeypatch.setattr("openkb.config.resolve_effective_config", lambda kb: ({}, {}))
    seen = []

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary updates must not invoke an audit")

    monkeypatch.setattr(compiler, "audit_candidates", forbidden)
    monkeypatch.setattr(compiler, "plan_audit_batches", forbidden)

    def sync(model, messages, step, **kwargs):
        if step == "concepts-plan":
            update = {"update": [{"name": "mosaic", "title": "Mosaic", "type": "product"}]}
            return json.dumps({"entities": update} if folder == "entities" else update)
        return current

    async def generate(model, messages, step, **kwargs):
        assert step != "source-attribution-audit"
        text = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
        assert prior in text  # Including original citations and source associations.
        assert "existing wiki pages are not evidence for new source facts" in text
        assert current in text
        seen.append(step)
        # Retain historical inline provenance, without promoting its old page-wide
        # audit footer to a certification of the newly generated combined page.
        return json.dumps({"content": historical + "\n" + current, "type": "product"})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", generate)
    await compiler.compile_long_doc("doc-b", summary, "id", tmp_path, "mock/model")
    result = page.read_text()
    assert historical in result and current in result
    assert "summaries/doc-a.md" in result and "summaries/doc-b.md" in result
    assert "Source attribution evidence:" not in result
    assert "[source doc-b p." not in result  # No fresh certification without audit.
    assert json.loads(source_path.read_text()) == [{"page": 1, "content": current}]
    assert seen == ["entity-update: mosaic" if folder == "entities" else "update: mosaic"]


@pytest.mark.asyncio
@pytest.mark.parametrize("long_doc,source_artifact", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("planned", [False, True])
async def test_default_publishes_without_any_audit_work(
    tmp_path, monkeypatch, long_doc, planned, source_artifact
):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    text = "Mosaic provides code review, code search, and IAM."
    if source_artifact:
        (wiki / "sources/report.json").write_text(json.dumps([{"page": 1, "content": text}]))
    summary = wiki / "summaries/report.md"
    summary.write_text(text)
    source = tmp_path / "source.txt"
    source.write_text(text)
    monkeypatch.setattr("openkb.config.resolve_effective_config", lambda kb: ({}, {}))

    def forbidden(*a, **kw):
        pytest.fail("default compilation must not perform audit planning, size discovery or calls")

    for name in (
        "audit_candidates",
        "plan_audit_batches",
        "audit_messages",
        "load_summary_artifact",
    ):
        monkeypatch.setattr(compiler, name, forbidden)

    def sync(model, messages, step, **kw):
        if step == "concepts-plan":
            return json.dumps(
                {"create": [{"name": "review", "title": "Review"}]} if planned else {}
            )
        if step in ("overview", "summary-rewrite"):
            return text + " [[concepts/ghost]]"
        return json.dumps({"description": "Generated description", "content": text})

    async def generate(model, messages, step, **kw):
        assert step != "source-attribution-audit"
        return json.dumps({"content": text + " [[concepts/ghost]]"})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", generate)
    if long_doc:
        await compiler.compile_long_doc("report", summary, "id", tmp_path, "mock/model")
    else:
        await compiler.compile_short_doc("report", source, tmp_path, "mock/model")
    assert text in summary.read_text()
    assert "report" in (wiki / "index.md").read_text()
    if planned:
        assert text in (wiki / "concepts/review.md").read_text()
    for path in wiki.rglob("*.md"):
        body = path.read_text()
        assert "[[concepts/ghost]]" not in body
        assert "[source report p." not in body
        assert "verified" not in body
        assert "supported" not in body
        assert "audited" not in body
