"""Coverage regressions for bounded source-attribution audit inputs (offline only)."""

import json

import pytest

from openkb.grounding import MAX_AUDIT_CHARS, audit_messages


def small_source():
    pages = {
        n: (f"Physical page {n}. " + "Storage history transactions. " * 55) for n in range(1, 14)
    }
    pages[2] += " Scion is version control; Antique is code review, alongside code search and IAM."
    pages[11] += " Scion has directory permissions; review tooling has approval constraints."
    return pages


def test_small_document_evidence_is_complete_and_shared_once_for_every_target():
    pages = small_source()
    candidates = {
        "summaries/paper": "Scion and Antique overview of the whole document.",
        "entities/antique": "Antique review surface",
        "entities/scion": "Scion directory permissions",
        "concepts/security": "Repository permissions and review approvals",
    }
    messages = audit_messages(pages, candidates, doc_name="paper")
    body = json.loads(messages[1]["content"])
    assert body["source_pages"] == {str(n): text for n, text in pages.items()}
    assert body["evidence_scope"] == "complete_original_document_shared_once"
    for target in candidates:
        assert body["candidate_pages"][target] == list(body["source_pages"])
    assert messages[1]["content"].count("Antique is code review") == 1
    assert messages[1]["content"].count("review tooling has approval constraints") == 1
    assert len(messages[0]["content"]) + len(messages[1]["content"]) <= MAX_AUDIT_CHARS
    assert "complete_original_document_shared_once" in messages[0]["content"]


@pytest.mark.asyncio
async def test_audit_accepts_checked_quote_from_shared_complete_source():
    from openkb.grounding import audit_candidates

    pages = small_source()
    quote = "Scion has directory permissions; review tooling has approval constraints."
    candidates = {"entities/scion": "Scion directory permissions"}
    calls = []

    async def audit(model, messages, step, **kwargs):
        calls.append(step)
        return json.dumps(
            {
                "targets": [
                    {
                        "target": "entities/scion",
                        "verdict": "supported",
                        "evidence": [{"doc": "paper", "page": 11, "quote": quote}],
                    }
                ]
            }
        )

    await audit_candidates(pages, candidates, "mock/model", audit, doc_name="paper")
    assert calls == ["source-attribution-audit"]
    assert "[source paper p. 11]" in candidates["entities/scion"]


def test_large_source_is_not_serialized_just_to_check_small_source_eligibility(monkeypatch):
    from openkb import grounding

    original = grounding.json.dumps

    def bounded_dumps(value, **kwargs):
        if isinstance(value, dict) and isinstance(value.get("1"), str):
            assert len(value["1"]) <= 32_000, "unbounded original serialized before size check"
        return original(value, **kwargs)

    monkeypatch.setattr(grounding.json, "dumps", bounded_dumps)
    messages = audit_messages({1: "Geology " * 12000}, {"entities/alpha": "Alpha"}, doc_name="book")
    assert "selected_original_page_windows_not_entire_pdf" in messages[1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["unclear", "unsupported"])
async def test_complete_source_does_not_turn_refusal_into_publication(verdict):
    from openkb.grounding import GroundingError, audit_candidates

    candidates = {"entities/scion": "Scion directory permissions"}
    before = dict(candidates)
    calls = []

    async def audit(model, messages, step, **kwargs):
        calls.append(step)
        return json.dumps(
            {
                "targets": [
                    {
                        "target": "entities/scion",
                        "verdict": verdict,
                        "evidence": [],
                    }
                ]
            }
        )

    with pytest.raises(GroundingError):
        await audit_candidates(small_source(), candidates, "mock/model", audit, doc_name="paper")
    assert candidates == before
    assert calls == ["source-attribution-audit"]


@pytest.mark.asyncio
async def test_complete_source_quote_still_requires_target_page_reference(monkeypatch):
    from openkb import grounding

    original = grounding.audit_messages

    def restricted(*args, **kwargs):
        messages = original(*args, **kwargs)
        body = json.loads(messages[1]["content"])
        body["candidate_pages"]["entities/scion"].remove("11")
        messages[1]["content"] = json.dumps(body)
        return messages

    monkeypatch.setattr(grounding, "audit_messages", restricted)
    candidates = {"entities/scion": "Scion directory permissions"}
    before = dict(candidates)
    quote = "Scion has directory permissions; review tooling has approval constraints."

    async def audit(*args, **kwargs):
        return json.dumps(
            {
                "targets": [
                    {
                        "target": "entities/scion",
                        "verdict": "supported",
                        "evidence": [
                            {
                                "doc": "paper",
                                "page": 11,
                                "quote": quote,
                            }
                        ],
                    }
                ]
            }
        )

    with pytest.raises(grounding.GroundingError, match="selected evidence"):
        await grounding.audit_candidates(
            small_source(), candidates, "mock/model", audit, doc_name="paper"
        )
    assert candidates == before


@pytest.mark.asyncio
async def test_oversized_complete_batch_is_not_downgraded_to_partial_context():
    from openkb.grounding import GroundingError, audit_candidates

    candidates = {"entities/scion": "Scion " + "x" * 80_000}
    before = dict(candidates)

    async def forbidden(*args, **kwargs):
        pytest.fail("an oversized complete batch must stop before a model call")

    with pytest.raises(GroundingError, match="bound"):
        await audit_candidates(
            small_source(), candidates, "mock/model", forbidden, doc_name="paper"
        )
    assert candidates == before


@pytest.mark.parametrize("pages", [{n: "short" for n in range(1, 34)}, {1: "\\" * 20_000}])
def test_complete_source_eligibility_checks_page_count_and_serialized_size(pages):
    body = json.loads(
        audit_messages(pages, {"entities/scion": "Scion"}, doc_name="paper")[1]["content"]
    )
    assert body["evidence_scope"] == "selected_original_page_windows_not_entire_pdf"


@pytest.mark.parametrize("page_count", [32, 33])
def test_exact_complete_source_page_limit(page_count):
    pages = {n: "Original text" for n in range(1, page_count + 1)}
    body = json.loads(audit_messages(pages, {"entities/alpha": "Alpha"})[1]["content"])
    assert (body["evidence_scope"] == "complete_original_document_shared_once") == (
        page_count == 32
    )


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_exact_complete_source_serialized_character_limit(delta):
    from openkb.grounding import MAX_COMPLETE_AUDIT_SOURCE_CHARS

    overhead = len(json.dumps({"1": ""}, ensure_ascii=False))
    pages = {1: "x" * (MAX_COMPLETE_AUDIT_SOURCE_CHARS - overhead + delta)}
    assert (
        len(json.dumps({"1": pages[1]}, ensure_ascii=False))
        == MAX_COMPLETE_AUDIT_SOURCE_CHARS + delta
    )
    body = json.loads(audit_messages(pages, {"entities/alpha": "Alpha"})[1]["content"])
    assert (body["evidence_scope"] == "complete_original_document_shared_once") == (delta <= 0)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_exact_complete_batch_character_limit(delta):
    from openkb.grounding import GroundingError

    pages = {1: "Original text"}
    initial = audit_messages(pages, {"entities/alpha": ""})
    used = sum(len(message["content"]) for message in initial)
    candidates = {"entities/alpha": "x" * (MAX_AUDIT_CHARS - used + delta)}
    if delta > 0:
        with pytest.raises(GroundingError, match="bound"):
            audit_messages(pages, candidates)
    else:
        messages = audit_messages(pages, candidates)
        assert sum(len(message["content"]) for message in messages) == MAX_AUDIT_CHARS + delta
