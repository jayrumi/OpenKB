"""Section audit structural tests: synthetic verdicts are not semantic proof."""

import json
import sqlite3

import pytest

from openkb import grounding
from openkb.summary_audit import digest, snapshot
from openkb.tree_renderer import render_summary_md


def book():
    tree = {
        "structure": [
            {
                "node_id": "root",
                "title": "Alpha",
                "start_index": 1,
                "end_index": 1,
                "summary": "Alpha review, search, annotations.",
                "nodes": [
                    {
                        "node_id": str(i),
                        "title": f"Part {i}",
                        "start_index": 1,
                        "end_index": 1,
                        "summary": "Detail " + "x" * 1800,
                    }
                    for i in range(60)
                ],
            }
        ]
    }
    pages = {1: "Alpha review, search, annotations."}
    text = "Description\n" + render_summary_md(tree, "book", "id")
    return tree, pages, {"summaries/book": text}


def test_oversized_verified_summary_is_section_aligned_exact_coverage():
    tree, pages, candidates = book()
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    assert artifact.target_digest == digest(candidates["summaries/book"])
    assert artifact.source_digest == digest(snapshot(pages))
    assert artifact.tree_digest == digest(snapshot(tree))
    batches = grounding.plan_audit_batches(
        pages, candidates, doc_name="book", summary_artifact=artifact
    )
    assert len(batches) == 61
    assert (
        b"".join(f.owned.encode() for f in artifact.fragments)
        == candidates["summaries/book"].encode()
    )
    seen = []
    for fragment, messages in zip(artifact.fragments, batches, strict=True):
        body = json.loads(messages[1]["content"])
        assert body["candidates"] == {fragment.key: fragment.context + fragment.owned}
        assert fragment.key.startswith("summaries/book#span:")
        assert fragment.context == ("" if not seen else artifact.preamble + artifact.sections[0])
        assert sum(len(m["content"]) for m in messages) <= 100_000
        seen.append(fragment.key)
    assert len(set(seen)) == 61


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal", [None, "unsupported", "unclear", "missing", "duplicate", "wrong"]
)
async def test_every_fragment_is_required_and_parent_is_annotated_only_after_all_pass(
    refusal, monkeypatch
):
    routed = []
    original_route = grounding.risky_targets

    def route(pages, candidates):
        routed.append(tuple(candidates))
        return original_route(pages, candidates)

    monkeypatch.setattr(grounding, "risky_targets", route)
    tree, pages, candidates = book()
    before = dict(candidates)
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    calls = []

    async def audit(model, messages, step, **kwargs):
        assert candidates == before
        body = json.loads(messages[1]["content"])
        calls.append(body)
        results = [
            {
                "target": key,
                "verdict": "supported",
                "evidence": [{"doc": "book", "page": 1, "quote": pages[1]}],
            }
            for key in body["candidates"]
        ]
        if len(calls) == 61 and refusal:
            if refusal == "missing":
                results = []
            elif refusal == "duplicate":
                results *= 2
            elif refusal == "wrong":
                results[0]["target"] = artifact.fragments[0].key
            else:
                results[0]["verdict"] = refusal
        return json.dumps({"targets": results})

    if refusal:
        with pytest.raises(grounding.GroundingError):
            await grounding.audit_candidates(
                pages, candidates, "mock", audit, doc_name="book", summary_artifact=artifact
            )
        assert candidates == before
    else:
        await grounding.audit_candidates(
            pages, candidates, "mock", audit, doc_name="book", summary_artifact=artifact
        )
        assert candidates[artifact.parent] == before[artifact.parent] + (
            "\n\nSource attribution evidence: [source book p. 1]\n"
        )
    assert len(calls) == 61
    assert routed == [("summaries/book",)]


@pytest.mark.asyncio
async def test_verified_oversized_parent_cannot_escape_required_audit_by_lexical_routing():
    tree, pages, candidates = book()
    pages = {1: "plain source without enumeration"}
    assert grounding.risky_targets(pages, candidates) == {}
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    calls = []

    async def refuse(*args, **kwargs):
        calls.append(True)
        return '{"targets": []}'

    with pytest.raises(grounding.GroundingError):
        await grounding.audit_candidates(
            pages, candidates, "mock", refuse, doc_name="book", summary_artifact=artifact
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_other_required_candidate_refusal_blocks_parent_suffix():
    tree, pages, candidates = book()
    candidates["entities/alpha"] = "Alpha"
    before = dict(candidates)
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    calls = 0

    async def audit(model, messages, step, **kwargs):
        nonlocal calls
        calls += 1
        assert candidates == before
        keys = json.loads(messages[1]["content"])["candidates"]
        return json.dumps(
            {
                "targets": [
                    {
                        "target": k,
                        "verdict": "unsupported" if k == "entities/alpha" else "supported",
                        "evidence": [{"doc": "book", "page": 1, "quote": pages[1]}],
                    }
                    for k in keys
                ]
            }
        )

    with pytest.raises(grounding.GroundingError):
        await grounding.audit_candidates(
            pages, candidates, "mock", audit, doc_name="book", summary_artifact=artifact
        )
    assert calls == 62
    assert candidates == before


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["other_candidate", "baseline", "source", "tree", "parent"])
async def test_changed_inputs_never_receive_audit_annotations(change):
    tree, pages, candidates = book()
    candidates["entities/alpha"] = "Alpha"
    previous = {"entities/alpha": "Alpha prior"}
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    calls = 0

    async def audit(model, messages, step, **kwargs):
        nonlocal calls
        calls += 1
        body = json.loads(messages[1]["content"])
        if calls == 62:
            if change == "other_candidate":
                candidates["entities/alpha"] += " changed"
            elif change == "baseline":
                previous["entities/alpha"] += " changed"
            elif change == "source":
                pages[2] = "new source"
            elif change == "tree":
                tree["structure"][0]["summary"] += " changed"
            else:
                candidates[artifact.parent] += " changed"
        return json.dumps(
            {
                "targets": [
                    {
                        "target": k,
                        "verdict": "supported",
                        "scope": "new_or_changed_claims",
                        "evidence": [{"doc": "book", "page": 1, "quote": pages[1]}],
                    }
                    for k in body["candidates"]
                ]
            }
        )

    with pytest.raises(grounding.GroundingError, match="changed"):
        await grounding.audit_candidates(
            pages,
            candidates,
            "mock",
            audit,
            doc_name="book",
            previous_versions=previous,
            summary_artifact=artifact,
        )
    assert calls == 62
    assert not any("Source attribution evidence:" in t for t in candidates.values())


def test_tampered_fragment_context_is_rejected():
    from dataclasses import replace

    tree, pages, candidates = book()
    artifact = grounding.freeze_summary(
        "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
    )
    altered = replace(artifact.fragments[-1], context="invented ancestor")
    artifact = replace(artifact, fragments=artifact.fragments[:-1] + (altered,))
    with pytest.raises(grounding.GroundingError):
        artifact.verify(pages, candidates)


@pytest.mark.parametrize("stored_change", ["source", "summary", "tree", "tree_type"])
def test_stored_frozen_inputs_are_rechecked_before_acceptance(tmp_path, stored_change):
    tree, pages, candidates = book()
    summary = tmp_path / "summary.md"
    source = tmp_path / "source.json"
    summary.write_text(candidates["summaries/book"])
    source.write_text(snapshot(pages))

    def read_tree():
        if tree.get("invalid"):
            raise TypeError("malformed stored tree")
        return tree

    artifact = grounding.freeze_summary(
        "summaries/book",
        candidates["summaries/book"],
        pages,
        tree,
        prefix="Description\n",
        tree_reader=read_tree,
        guards=((summary, digest(summary.read_text())), (source, digest(source.read_text()))),
    )
    if stored_change == "source":
        source.write_text("changed")
    elif stored_change == "summary":
        summary.write_text("changed")
    elif stored_change == "tree":
        tree["structure"][0]["summary"] = "changed"
    else:
        tree["invalid"] = True
    with pytest.raises(grounding.GroundingError):
        artifact.verify(pages, candidates)


@pytest.mark.parametrize("invalid", [None, "bad", [None], [{"nodes": None}]])
def test_malformed_tree_is_typed_terminal_refusal(invalid):
    tree, pages, candidates = book()
    tree["structure"] = invalid
    with pytest.raises(grounding.GroundingError):
        grounding.freeze_summary(
            "summaries/book", candidates["summaries/book"], pages, tree, prefix="Description\n"
        )


@pytest.mark.parametrize("kind", ["baseline", "stale", "ambiguous", "missing", "duplicate"])
def test_invalid_summary_fragmentation_fails_locally(kind):
    from dataclasses import replace

    tree, pages, candidates = book()
    parent = "summaries/book"
    if kind == "stale":
        tree["structure"][0]["title"] = "wrong"
    if kind == "ambiguous":
        tree["structure"][0]["summary"] += "\n# fake (pages 1–1)\n"
        candidates[parent] = "Description\n" + render_summary_md(tree, "book", "id")
    if kind in ("stale", "ambiguous"):
        with pytest.raises(grounding.GroundingError):
            grounding.freeze_summary(
                parent, candidates[parent], pages, tree, prefix="Description\n"
            )
        return
    artifact = grounding.freeze_summary(
        parent, candidates[parent], pages, tree, prefix="Description\n"
    )
    if kind == "missing":
        artifact = replace(artifact, fragments=artifact.fragments[:-1])
    if kind == "duplicate":
        artifact = replace(artifact, fragments=artifact.fragments + artifact.fragments[-1:])
    with pytest.raises(grounding.GroundingError):
        grounding.plan_audit_batches(
            pages,
            candidates,
            doc_name="book",
            summary_artifact=artifact,
            previous_versions={parent: "legacy"} if kind == "baseline" else {},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_plan", [True, False])
@pytest.mark.parametrize("bounded_parent", [True, False])
async def test_compiler_long_summary_refusal_preserves_all_wiki_bytes(
    kb_dir, monkeypatch, empty_plan, bounded_parent, enable_grounding_audit
):
    from openkb.agent import compiler

    tree, pages, candidates = book()
    if bounded_parent:
        for child in tree["structure"][0]["nodes"]:
            child["summary"] = "Detail " + "x" * 1500
        candidates["summaries/book"] = "Description\n" + render_summary_md(tree, "book", "id")
        pages[1] += " " + "x" * 11_000
        assert len(candidates["summaries/book"]) < 100_000
        with pytest.raises(grounding.AuditSizeError):
            grounding.audit_messages(pages, candidates, doc_name="book")
    summary_path = kb_dir / "wiki/summaries/book.md"
    summary_path.write_text(candidates["summaries/book"].removeprefix("Description\n"))
    (kb_dir / "wiki/sources/book.json").write_text(
        json.dumps([{"page": n, "content": t} for n, t in pages.items()])
    )
    with sqlite3.connect(kb_dir / ".openkb/pageindex.db") as conn:
        conn.execute(
            "CREATE TABLE documents (doc_id TEXT, collection_name TEXT, "
            "doc_name TEXT, doc_description TEXT, structure TEXT)"
        )
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            ("id", "default", "book.pdf", "Description", json.dumps(tree["structure"])),
        )
    plan = {} if empty_plan else {"create": [{"name": "alpha", "title": "Alpha"}]}

    def generate(model, messages, step, **kwargs):
        return {
            "overview": "Alpha overview",
            "concepts-plan": json.dumps(plan),
            "concept-create:alpha": json.dumps({"description": "Alpha", "content": "Alpha"}),
        }.get(step, json.dumps({"description": "Alpha", "content": "Alpha"}))

    calls = []

    async def audit(model, messages, step, **kwargs):
        if step != "source-attribution-audit":
            return json.dumps({"description": "Alpha", "content": "Alpha"})
        body = json.loads(messages[1]["content"])
        calls.append(body)
        results = [
            {
                "target": key,
                "verdict": "supported",
                "evidence": [{"doc": "book", "page": 1, "quote": pages[1]}],
            }
            for key in body["candidates"]
        ]
        if (
            any(key.startswith("summaries/book#span:") for key in body["candidates"])
            and len(
                [
                    b
                    for b in calls
                    if any(k.startswith("summaries/book#span:") for k in b["candidates"])
                ]
            )
            == 61
        ):
            results[0]["verdict"] = "unsupported"
        return json.dumps({"targets": results})

    monkeypatch.setattr(compiler, "_llm_call", generate)
    monkeypatch.setattr(compiler, "_llm_call_async", audit)
    before = {p: p.read_bytes() for p in (kb_dir / "wiki").rglob("*") if p.is_file()}
    with pytest.raises(grounding.GroundingError):
        await compiler.compile_long_doc("book", summary_path, "id", kb_dir, "mock", "Description")
    assert len(calls) == 61
    assert {p: p.read_bytes() for p in (kb_dir / "wiki").rglob("*") if p.is_file()} == before
