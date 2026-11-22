"""Bounded evidence retrieval regressions; no network or model inference."""

import json

import pytest

from openkb.agent import compiler


@pytest.mark.asyncio
async def test_same_physical_page_different_windows_cannot_cross_candidate_evidence():
    from openkb.grounding import GroundingError, audit_candidates

    alpha = "Alpha provides review, search, and annotations."
    beta = "Beta provides storage, indexing, and snapshots."
    pages = {2: alpha + " " + "unrelated " * 4000 + beta}
    candidates = {"entities/alpha": "Alpha review", "entities/beta": "Beta storage"}

    async def audit(model, messages, step, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert alpha in payload["candidate_pages"]["entities/alpha"]["2"]
        assert beta not in payload["candidate_pages"]["entities/alpha"]["2"]
        return json.dumps(
            {
                "targets": [
                    {
                        "target": key,
                        "verdict": "supported",
                        "evidence": [{"doc": "book", "page": 2, "quote": beta}],
                    }
                    for key in payload["candidates"]
                ]
            }
        )

    before = dict(candidates)
    with pytest.raises(GroundingError, match="selected evidence"):
        await audit_candidates(pages, candidates, "mock/model", audit, doc_name="book")
    assert candidates == before


def test_cited_late_page_is_selected_despite_zero_lexical_overlap():
    from openkb.grounding import select_source_pages

    pages = {n: "unrelated geology " * 400 for n in range(1, 659)}
    pages[658] = "XYZ: original physical evidence."
    chosen = select_source_pages(pages, "Mosaic [source book p. 658]", doc_name="book")
    assert chosen[658] == pages[658]
    assert sum(len(v) for v in chosen.values()) <= 12000


@pytest.mark.asyncio
async def test_large_book_uses_bounded_target_evidence_in_one_audit(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    evidence = "Zircon provides review, search, and annotations."
    pages = [{"page": n, "content": ("Unrelated geology chapter. " * 200)} for n in range(1, 659)]
    pages[-1]["content"] += evidence
    raw = json.dumps(pages)
    (wiki / "sources/book.json").write_text(raw)
    summary = wiki / "summaries/book.md"
    summary.write_text("Zircon suite overview")
    calls = []

    def sync(model, messages, step, **kwargs):
        assert len(json.dumps(messages)) < 60_000, "whole book leaked into common context"
        calls.append(step)
        if step == "concepts-plan":
            return json.dumps(
                {"entities": {"create": [{"name": "zircon", "title": "Zircon", "type": "product"}]}}
            )
        return "Zircon suite overview"

    async def call(model, messages, step, **kwargs):
        assert len(json.dumps(messages)) < 100_000
        calls.append(step)
        if step == "source-attribution-audit":
            payload = json.loads(messages[1]["content"])
            assert payload["current_document"] == "book"
            assert payload["evidence_scope"] == "selected_original_page_windows_not_entire_pdf"
            assert evidence in str(payload["source_pages"])
            assert "658" in payload["source_pages"]
            assert set(payload["candidate_pages"]) == set(payload["candidates"])
            return json.dumps(
                {
                    "targets": [
                        {
                            "target": key,
                            "verdict": "supported",
                            "evidence": [{"doc": "book", "page": 658, "quote": evidence}],
                        }
                        for key in payload["candidates"]
                    ]
                }
            )
        assert evidence in str(messages), "late original evidence must reach per-target generation"
        assert "[source book p. 658]" in str(messages)
        return json.dumps({"description": "Zircon suite", "content": evidence, "type": "product"})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", call)
    await compiler.compile_long_doc("book", summary, "id", tmp_path, "mock/model")
    assert calls == ["overview", "concepts-plan", "entity: zircon", "source-attribution-audit"]
    assert "[source book p. 658]" in (wiki / "entities/zircon.md").read_text()
    assert (wiki / "sources/book.json").read_text() == raw


@pytest.mark.asyncio
async def test_oversized_previous_version_blocks_before_candidate_generation(tmp_path, monkeypatch):
    from openkb.grounding import GroundingError

    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    summary = wiki / "summaries/book.md"
    summary.write_text("Zircon overview")
    (wiki / "entities/zircon.md").write_text("Prior fact " * 11000)
    calls = []

    def sync(model, messages, step, **kwargs):
        calls.append(step)
        if step == "concepts-plan":
            return json.dumps({"entities": {"update": [{"name": "zircon"}]}})
        return "Zircon overview"

    async def forbidden(*args, **kwargs):
        pytest.fail("known oversized baseline must fail before costly generation")

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", forbidden)
    with pytest.raises(GroundingError, match="bound"):
        await compiler.compile_long_doc("book", summary, "id", tmp_path, "mock/model")
    assert calls == ["overview", "concepts-plan"]
