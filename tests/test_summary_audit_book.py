"""Opt-in actual saved-book structural proof. NO semantic support claim."""

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest

from openkb import grounding
from openkb.summary_audit import digest, load_summary_artifact, snapshot


@pytest.mark.asyncio
async def test_actual_saved_book_section_audit_proof(monkeypatch):
    root = os.environ.get("OPENKB_SECTION_AUDIT_BOOK")
    if not root:
        pytest.skip("set OPENKB_SECTION_AUDIT_BOOK to the retained read-only recovered KB")

    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in saved-book proof")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    kb = Path(root)
    doc = "Instrumentariy_hraneniya_i_analiza_dannyh"
    summary = kb / f"wiki/summaries/{doc}.md"
    source = kb / f"wiki/sources/{doc}.json"
    db = kb / ".openkb/pageindex.db"
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary, source, db)}
    original = summary.read_bytes()
    assert len(original) == 1896645
    assert (
        hashlib.sha256(original).hexdigest()
        == "cf8a0c4a8948c65fb93fdf34de36fb3ac937fdd80b2b697d2aae5f1d273ba5b3"
    )
    pages = grounding.load_source_pages(kb / "wiki", doc)
    assert len(pages) == 648
    # Read the stored description, not a generated ancestor, as candidate context.
    import sqlite3

    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        prefix = (
            conn.execute(
                "SELECT doc_description FROM documents WHERE doc_id = ?",
                ("49d41265-4b10-449c-87bd-54ef0ada15c4",),
            ).fetchone()[0]
            + "\n"
        )
    artifact = load_summary_artifact(
        kb, summary, "49d41265-4b10-449c-87bd-54ef0ada15c4", doc, pages, prefix=prefix
    )
    candidates = {artifact.parent: artifact.text}
    assert len(artifact.fragments) == 949
    assert b"".join(f.owned.encode() for f in artifact.fragments) == prefix.encode() + original
    assert grounding.risky_targets(pages, candidates) == candidates
    # Cross-check against the independently saved pre-implementation probe map.
    manifest_path = kb.parents[1] / "summary-audit-feasibility/sections.json"
    protected[manifest_path] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 949
    raw_text = original.decode()
    expected_sections = {}
    for row in manifest:
        start, end = row["char_span"]
        own = raw_text[start:end]
        assert digest(own) == row["sha256"]
        assert original[slice(*row["byte_span"])] == own.encode()
        expected_sections[row["node_id"]] = own
    preamble = prefix + raw_text[: manifest[0]["char_span"][0]]
    for i, (row, fragment) in enumerate(zip(manifest, artifact.fragments, strict=True)):
        assert fragment.start == (len(prefix) + row["char_span"][0] if i else 0)
        assert fragment.end == len(prefix) + row["char_span"][1]
        assert fragment.context == (
            preamble + "".join(expected_sections[a] for a in row["ancestor_ids"]) if i else ""
        )
    batches = grounding.plan_audit_batches(
        pages, candidates, doc_name=doc, summary_artifact=artifact
    )
    assert len(batches) == 949
    requests = []
    for fragment, messages in zip(artifact.fragments, batches, strict=True):
        body = json.loads(messages[1]["content"])
        assert body["candidates"] == {fragment.key: fragment.context + fragment.owned}
        assert body["previous_versions"] == {}
        assert body["evidence_scope"] == "selected_original_page_windows_not_entire_pdf"
        windows = body["candidate_pages"][fragment.key]
        evidence_chars = sum(len(v) + 80 for v in windows.values())
        request_chars = sum(len(m["content"]) for m in messages)
        assert evidence_chars <= 12_000
        assert request_chars <= 100_000
        for n, window in windows.items():
            for part in window.split("\n[... omitted ...]\n"):
                assert part in pages[int(n)]
        requests.append(
            {
                "key": fragment.key,
                "start": fragment.start,
                "end": fragment.end,
                "owned_sha256": digest(fragment.owned),
                "context_sha256": digest(fragment.context),
                "request_chars": request_chars,
                "evidence_chars": evidence_chars,
                "selected_pages": list(windows),
            }
        )
    # Reuse the EXACT locally planned requests for the two synthetic verdict runs.
    # This saves repeated lexical scans, not semantic calls (there are none).
    monkeypatch.setattr(grounding, "plan_audit_batches", lambda *a, **kw: batches)
    calls = 0
    reject_last = True
    before = dict(candidates)

    async def synthetic(model, messages, step, **kwargs):
        nonlocal calls
        calls += 1
        assert candidates == before
        body = json.loads(messages[1]["content"])
        key = next(iter(body["candidates"]))
        n, window = next(iter(body["candidate_pages"][key].items()))
        return json.dumps(
            {
                "targets": [
                    {
                        "target": key,
                        "verdict": "unsupported" if reject_last and calls == 949 else "supported",
                        "evidence": [
                            {
                                "doc": doc,
                                "page": int(n),
                                "quote": window.split("\n[... omitted ...]\n")[0],
                            }
                        ],
                    }
                ]
            }
        )

    with pytest.raises(grounding.GroundingError):
        await grounding.audit_candidates(
            pages, candidates, "offline", synthetic, doc_name=doc, summary_artifact=artifact
        )
    assert calls == 949 and candidates == before
    calls = 0
    reject_last = False
    await grounding.audit_candidates(
        pages, candidates, "offline", synthetic, doc_name=doc, summary_artifact=artifact
    )
    assert calls == 949
    assert candidates[artifact.parent].startswith(
        artifact.text + "\n\nSource attribution evidence:"
    )
    assert candidates[artifact.parent].count("Source attribution evidence:") == 1
    body, suffix = candidates[artifact.parent].split("\n\nSource attribution evidence:")
    assert body.encode() == prefix.encode() + original
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected} == protected
    report = {
        "semantic_audit": False,
        "network_forbidden": True,
        "inference_calls": 0,
        "sections": 949,
        "summary_bytes": len(original),
        "summary_sha256": protected[summary],
        "target_sha256": digest(artifact.text),
        "tree_sha256": digest(artifact.tree_snapshot),
        "source_sha256": protected[source],
        "source_pages_digest": digest(snapshot(pages)),
        "exact_once_contiguous_utf8_reassembly": True,
        "ancestor_context_crosschecked_against_saved_probe": True,
        "final_refusal_zero_mutations": True,
        "one_evidence_suffix_unchanged_original_body": True,
        "protected_inputs_unchanged": True,
        "requests": requests,
        "min_request_chars": min(r["request_chars"] for r in requests),
        "max_request_chars": max(r["request_chars"] for r in requests),
        "max_evidence_chars": max(r["evidence_chars"] for r in requests),
    }
    output = os.environ.get("OPENKB_SECTION_AUDIT_PROOF")
    if output:
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=2))
