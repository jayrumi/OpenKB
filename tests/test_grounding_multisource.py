"""Offline update regressions: semantic responses are stubs, not truth proofs."""

import asyncio
import json

import pytest

from openkb.agent import compiler


@pytest.mark.parametrize("evidence_doc", ["doc-a", "doc-b", None])
def test_physical_page_requires_current_document_identity(evidence_doc):
    from openkb.grounding import GroundingError, validate_audit

    item = {"page": 2, "quote": "same text in two PDFs"}
    if evidence_doc is not None:
        item["doc"] = evidence_doc
    raw = json.dumps(
        {"targets": [{"target": "entities/mosaic", "verdict": "supported", "evidence": [item]}]}
    )
    if evidence_doc == "doc-b":
        assert validate_audit(
            raw, {"entities/mosaic": "claim"}, {2: item["quote"]}, doc_name="doc-b"
        )
    else:
        with pytest.raises(GroundingError):
            validate_audit(raw, {"entities/mosaic": "claim"}, {2: item["quote"]}, doc_name="doc-b")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["supported", "unsupported", "cancel"])
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_update_audits_exact_previous_version_without_trusting_new_claims(
    tmp_path, monkeypatch, outcome
):
    wiki = tmp_path / "wiki"
    for folder in ("sources", "summaries", "entities"):
        (wiki / folder).mkdir(parents=True)
    prior = (
        "---\nsources: [summaries/doc-a.md]\ntype: product\n"
        "description: Mosaic founded in 2001\n---\n"
        "Mosaic was founded in 2001. [source doc-a p. 2]\n"
    )
    (wiki / "entities/mosaic.md").write_text(prior)
    current = "Mosaic provides review, search, and annotations."
    (wiki / "sources/doc-b.json").write_text(json.dumps([{"page": 2, "content": current}]))
    (wiki / "summaries/doc-b.md").write_text(current)
    before = {str(p): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    seen = []

    def sync(model, messages, step, **kwargs):
        if step == "concepts-plan":
            return json.dumps(
                {"entities": {"update": [{"name": "mosaic", "title": "Mosaic", "type": "product"}]}}
            )
        return current

    async def call(model, messages, step, **kwargs):
        if step == "source-attribution-audit":
            payload = json.loads(messages[1]["content"])
            assert payload["previous_versions"]["entities/mosaic"] == prior
            assert "unchanged" in messages[0]["content"].lower()
            assert "new or changed" in messages[0]["content"].lower()
            seen.append(step)
            if outcome == "cancel":
                raise asyncio.CancelledError
            return json.dumps(
                {
                    "targets": [
                        {
                            "target": key,
                            "verdict": outcome,
                            "scope": "new_or_changed_claims",
                            "evidence": [{"doc": "doc-b", "page": 2, "quote": current}],
                        }
                        for key in payload["candidates"]
                    ]
                }
            )
        assert any(prior in m["content"] for m in messages if isinstance(m.get("content"), str)), (
            "generation must also see previous source associations"
        )
        content = "Mosaic was founded in 2001. [source doc-a p. 2]\n" + current
        if outcome == "unsupported":
            content += " Mosaic also owns IAM."
        return json.dumps({"description": "Mosaic review", "content": content, "type": "product"})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", call)
    if outcome == "supported":
        await compiler.compile_long_doc(
            "doc-b", wiki / "summaries/doc-b.md", "id", tmp_path, "mock/model"
        )
        result = (wiki / "entities/mosaic.md").read_text()
        assert "founded in 2001" in result
        assert "summaries/doc-a.md" in result and "summaries/doc-b.md" in result
        assert "[source doc-a p. 2]" in result and "[source doc-b p. 2]" in result
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancel" else RuntimeError):
            await compiler.compile_long_doc(
                "doc-b", wiki / "summaries/doc-b.md", "id", tmp_path, "mock/model"
            )
        assert before == {str(p): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    assert seen == ["source-attribution-audit"]
