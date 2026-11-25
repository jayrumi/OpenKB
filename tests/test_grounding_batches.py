"""Deterministic local audit batching; actual private capture is opt-in."""

import json
import os
from pathlib import Path

import pytest

from openkb import grounding


@pytest.fixture
def actual_input():
    path = os.environ.get("OPENKB_AUDIT_CAPTURE")
    if not path:
        pytest.skip("set OPENKB_AUDIT_CAPTURE to the retained actual audit input")
    data = json.loads(Path(path).read_text())
    data["source_pages"] = {int(k): v for k, v in data["source_pages"].items()}
    return data


def test_actual_overview_and_eight_targets_fit_bounded_batches(actual_input):
    data = actual_input
    assert len(data["targets"]) == 8
    assert len(data["source_pages"]) == 648
    batches = grounding.plan_audit_batches(
        data["source_pages"],
        data["targets"],
        doc_name=data["doc_name"],
        previous_versions=data["previous_versions"],
    )
    assert len(batches) > 1
    collected = {}
    for messages in batches:
        assert sum(len(m["content"]) for m in messages) <= grounding.MAX_AUDIT_CHARS
        body = json.loads(messages[1]["content"])
        assert not collected.keys() & body["candidates"].keys()
        collected.update(body["candidates"])
        for target, windows in body["candidate_pages"].items():
            assert sum(len(v) for v in windows.values()) <= grounding.MAX_EVIDENCE_CHARS
            if target.startswith("summaries/"):
                assert {"524", "603", "642", "651", "656"} <= windows.keys()
    assert collected == data["targets"]


def test_small_source_batches_keep_complete_pages_and_exact_baselines():
    pages = {1: "Alpha provides review, search, and annotations."}
    targets = {f"entities/alpha{i}": "Alpha" + "x" * 35_000 for i in range(3)}
    previous = {key: f"---\nsources: [old]\n---\nExact prior {key}\n" for key in targets}
    batches = grounding.plan_audit_batches(
        pages, targets, doc_name="book", previous_versions={**previous, "unselected": "omit"}
    )
    assert batches == grounding.plan_audit_batches(
        pages, targets, doc_name="book", previous_versions=previous
    )
    assert len(batches) > 1
    seen = {}
    baselines = {}
    for messages in batches:
        body = json.loads(messages[1]["content"])
        assert body["evidence_scope"] == "complete_original_document_shared_once"
        assert body["source_pages"] == {"1": pages[1]}
        assert body["previous_versions"] == {key: previous[key] for key in body["candidates"]}
        seen.update(body["candidates"])
        baselines.update(body["previous_versions"])
    assert seen == targets
    assert baselines == previous


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["unsupported", "unclear", "malformed", "missing", "scope"])
async def test_later_batch_refusal_is_all_or_nothing_and_never_retried(refusal):
    pages = {1: "Alpha provides review, search, and annotations."}
    candidates = {f"entities/alpha{i}": "Alpha " + "x" * 55_000 for i in range(3)}
    before = dict(candidates)
    previous = {key: "---\nsources: [legacy]\n---\nUnchanged prior fact\n" for key in candidates}
    calls = []

    async def audit(model, messages, step, **kwargs):
        assert candidates == before, "earlier batch must not mutate candidates"
        body = json.loads(messages[1]["content"])
        calls.append(body)
        results = [
            {
                "target": key,
                "verdict": "supported",
                "scope": "new_or_changed_claims",
                "evidence": [{"doc": "book", "page": 1, "quote": pages[1]}],
            }
            for key in body["candidates"]
        ]
        if len(calls) == 2:
            if refusal == "malformed":
                return "not JSON"
            if refusal == "missing":
                results = []
            elif refusal == "scope":
                results[0].pop("scope")
            else:
                results[0]["verdict"] = refusal
        return json.dumps({"targets": results})

    with pytest.raises(grounding.GroundingError):
        await grounding.audit_candidates(
            pages, candidates, "mock", audit, doc_name="book", previous_versions=previous
        )
    assert len(calls) == 2
    assert candidates == before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["candidate", "baseline", "cited_evidence"])
async def test_single_oversized_target_fails_before_any_call(kind):
    pages = {n: "Alpha provides review, search, and annotations. " * 100 for n in range(1, 34)}
    text = "Alpha "
    previous = {}
    if kind == "candidate":
        text += "x" * 100_000
    elif kind == "baseline":
        previous = {"entities/alpha": "x" * 100_000}
    else:
        text += " ".join(f"[source book p. {n}]" for n in range(1, 10))
    candidates = {"entities/alpha": text}

    async def forbidden(*args, **kwargs):
        pytest.fail("single oversized target cannot call the model")

    with pytest.raises(grounding.GroundingError, match="bound"):
        await grounding.audit_candidates(
            pages, candidates, "mock", forbidden, doc_name="book", previous_versions=previous
        )
    assert candidates == {"entities/alpha": text}


def test_non_size_grounding_failure_is_not_split(monkeypatch):
    calls = []

    def missing(*args, **kwargs):
        calls.append(args)
        raise grounding.GroundingError("Grounding cited source page missing")

    monkeypatch.setattr(grounding, "audit_messages", missing)
    with pytest.raises(grounding.GroundingError, match="missing"):
        grounding.plan_audit_batches({1: "source"}, {"a": "A", "b": "B"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_actual_audit_candidates_uses_same_planner_and_complete_target_coverage(
    actual_input, monkeypatch
):
    data = actual_input
    expected = grounding.plan_audit_batches(
        data["source_pages"],
        data["targets"],
        doc_name=data["doc_name"],
        previous_versions=data["previous_versions"],
    )
    # This regression exercises all preflight-required targets without relying on
    # the intentionally selective lexical routing heuristic.
    monkeypatch.setattr(grounding, "risky_targets", lambda pages, candidates: dict(candidates))
    calls = []
    candidates = dict(data["targets"])

    async def audit(model, messages, step, **kwargs):
        calls.append(messages)
        assert candidates == data["targets"]
        body = json.loads(messages[1]["content"])
        results = []
        for key, windows in body["candidate_pages"].items():
            page, window = next(iter(windows.items()))
            quote = window.split("\n[... omitted ...]\n")[0]
            assert quote in data["source_pages"][int(page)]
            results.append(
                {
                    "target": key,
                    "verdict": "supported",
                    "evidence": [{"doc": data["doc_name"], "page": int(page), "quote": quote}],
                }
            )
        # Synthetic acceptance here tests plumbing only, not semantic support.
        return json.dumps({"targets": results})

    await grounding.audit_candidates(
        data["source_pages"],
        candidates,
        "mock",
        audit,
        doc_name=data["doc_name"],
        previous_versions=data["previous_versions"],
    )
    assert calls == expected
    assert all("Source attribution evidence:" in text for text in candidates.values())


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_compiler_preflight_replays_actual_cached_overview_plan_binding(
    actual_input, tmp_path, monkeypatch
):
    from openkb.agent import compiler

    capture = Path(os.environ["OPENKB_AUDIT_CAPTURE"]).parent
    overview = json.loads((capture / "0001-response.json").read_text())
    plan = json.loads((capture / "0002-response.json").read_text())
    assert overview["stage"] == "overview"
    assert plan["stage"] == "concepts-plan"
    data = actual_input
    summary_target = data["targets"][f"summaries/{data['doc_name']}"]
    assert summary_target.endswith(overview["returned_text"])
    prefix = summary_target[: -len(overview["returned_text"])]
    assert prefix.endswith("\n")
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    summary = wiki / "summaries" / f"{data['doc_name']}.md"
    summary.write_text("Unpublished local replay input")
    (wiki / "sources" / f"{data['doc_name']}.json").write_text(
        json.dumps([{"page": n, "content": text} for n, text in data["source_pages"].items()])
    )
    calls = []

    def cached(model, messages, step, **kwargs):
        calls.append(step)
        assert step in ("overview", "concepts-plan")
        return (overview if step == "overview" else plan)["returned_text"]

    class PreflightComplete(Exception):
        pass

    def preflight(pages, targets, **kwargs):
        assert pages == data["source_pages"]
        assert targets == data["targets"]
        assert kwargs == {"doc_name": data["doc_name"], "previous_versions": {}}
        assert len(grounding.plan_audit_batches(pages, targets, **kwargs)) > 1
        raise PreflightComplete

    async def forbidden(*args, **kwargs):
        pytest.fail("replay must stop after local preflight, before generation")

    monkeypatch.setattr(compiler, "_llm_call", cached)
    monkeypatch.setattr(compiler, "_llm_call_async", forbidden)
    monkeypatch.setattr(compiler, "plan_audit_batches", preflight)
    with pytest.raises(PreflightComplete):
        await compiler.compile_long_doc(
            data["doc_name"],
            summary,
            "offline-id",
            tmp_path,
            "mock",
            doc_description=prefix[:-1],
        )
    assert calls == ["overview", "concepts-plan"]
    assert summary.read_text() == "Unpublished local replay input"
    assert not (wiki / "index.md").exists()
    assert not list((wiki / "concepts").glob("*.md"))
    assert not list((wiki / "entities").glob("*.md"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_compiler_later_batch_refusal_publishes_nothing(tmp_path, monkeypatch):
    from openkb.agent import compiler

    wiki = tmp_path / "wiki"
    for folder in ("sources", "summaries", "entities"):
        (wiki / folder).mkdir(parents=True)
    quote = "Alpha provides review, search, and annotations."
    (wiki / "sources/book.json").write_text(json.dumps([{"page": 1, "content": quote}]))
    summary = wiki / "summaries/book.md"
    summary.write_text("---\ntitle: Original\n---\nAlpha original overview")
    (wiki / "index.md").write_text("Original index\n")
    for n in range(3):
        (wiki / f"entities/alpha{n}.md").write_text(
            "---\ntitle: Alpha\ntype: product\nsources: [old]\n---\nAlpha " + "x" * 55_000
        )
    before = {p.relative_to(wiki): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    audit_calls = []

    def sync(model, messages, step, **kwargs):
        if step == "concepts-plan":
            return json.dumps({"entities": {"update": [{"name": f"alpha{n}"} for n in range(3)]}})
        return "Alpha overview"

    async def call(model, messages, step, **kwargs):
        if step != "source-attribution-audit":
            return json.dumps({"description": "Alpha", "content": quote, "type": "product"})
        body = json.loads(messages[1]["content"])
        audit_calls.append(body)
        for key, baseline in body["previous_versions"].items():
            assert baseline == before[Path(key + ".md")].decode()
        return json.dumps(
            {
                "targets": [
                    {
                        "target": key,
                        "verdict": "supported" if len(audit_calls) == 1 else "unclear",
                        "scope": "new_or_changed_claims",
                        "evidence": [{"doc": "book", "page": 1, "quote": quote}],
                    }
                    for key in body["candidates"]
                ]
            }
        )

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", call)
    with pytest.raises(grounding.GroundingError):
        await compiler.compile_long_doc("book", summary, "offline-id", tmp_path, "mock")
    assert len(audit_calls) == 2
    after = {p.relative_to(wiki): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    assert after == before
