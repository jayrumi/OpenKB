"""Offline source-attribution regressions; all model responses are explicit stubs."""

import json

import pytest

from openkb.agent import compiler

SOURCE = "Our suite includes a review surface (called Mosaic), code search, and an IAM system."


@pytest.mark.asyncio
@pytest.mark.parametrize("long_doc", [False, True])
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_unsupported_attribution_aborts_before_any_wiki_write(
    tmp_path, monkeypatch, long_doc
):
    wiki = tmp_path / "wiki"
    for folder in ("concepts", "entities", "summaries", "sources"):
        (wiki / folder).mkdir(parents=True)
    (wiki / "index.md").write_text("old index")
    summary = wiki / "summaries/report.md"
    summary.write_text("old summary")
    old = wiki / "entities/mosaic.md"
    old.write_text("old entity")
    (wiki / "sources/report.json").write_text(
        json.dumps(
            [
                {"page": 1, "content": SOURCE},
                {"page": 2, "content": "Separate IAM supports directory access."},
            ]
        )
    )
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    before = {str(p): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    calls = []

    def sync(model, messages, step_name, **kwargs):
        if step_name == "concepts-plan":
            return json.dumps(
                {"entities": {"update": [{"name": "mosaic", "title": "Mosaic", "type": "product"}]}}
            )
        if step_name == "summary-rewrite":
            return "Mosaic provides review, search, and access control."
        return json.dumps({"content": "Mosaic provides review, search, and access control."})

    async def async_call(model, messages, step_name, **kwargs):
        calls.append(step_name)
        if step_name == "source-attribution-audit":
            return json.dumps(
                {
                    "targets": [
                        {"target": "entities/mosaic", "verdict": "unsupported", "evidence": []},
                        {"target": "summaries/report", "verdict": "unsupported", "evidence": []},
                    ]
                }
            )
        return json.dumps(
            {"content": "Mosaic provides review, search, and access control.", "type": "product"}
        )

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", async_call)
    with pytest.raises(RuntimeError, match="[Gg]round|attribution"):
        if long_doc:
            await compiler.compile_long_doc("report", summary, "id", tmp_path, "mock/model")
        else:
            await compiler.compile_short_doc("report", source, tmp_path, "mock/model")
    assert before == {str(p): p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    assert calls.count("source-attribution-audit") == 1


@pytest.mark.asyncio
async def test_single_bounded_batch_accepts_real_combined_product_and_retains_pages():
    from openkb.grounding import audit_candidates

    source = "Prism provides code review, code search, and directory access."
    pages = {n: f"Physical page {n}" for n in range(1, 14)}
    pages[2] = source
    candidates = {
        f"entities/product-{n}": "Prism provides all three capabilities." for n in range(12)
    }
    calls = []

    async def audit(model, messages, step, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert payload["source_pages"]["2"] == source
        assert len(payload["source_pages"]) == 13
        calls.append(step)
        return json.dumps(
            {
                "targets": [
                    {
                        "target": key,
                        "verdict": "supported",
                        "evidence": [{"page": 2, "quote": source}],
                    }
                    for key in payload["candidates"]
                ]
            }
        )

    await audit_candidates(pages, candidates, "mock/model", audit)
    assert calls == ["source-attribution-audit"]
    assert all("[source p. 2]" in text for text in candidates.values())


@pytest.mark.asyncio
async def test_no_audit_without_source_enumeration():
    from openkb.grounding import audit_candidates

    async def forbidden(*args, **kwargs):
        pytest.fail("ordinary documents must not acquire hidden model calls")

    await audit_candidates(
        {1: "Mosaic is a review tool. Slate is another tool."},
        {"entities/mosaic": "Mosaic reviews code."},
        "mock/model",
        forbidden,
    )
    await audit_candidates(
        {},
        {"summaries/report": "---\nsources: [raw/report.pdf]\n---\nIndexed summary"},
        "mock/model",
        forbidden,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_page",
        "footer_page",
        "wrong_quote",
        "missing",
        "extra",
        "duplicate",
        "unclear",
        "scalar",
        "fenced",
        "extra_field",
    ],
)
def test_audit_response_is_closed_and_quotes_must_match_same_physical_page(damage):
    from openkb.grounding import GroundingError, validate_audit

    pages = {n: f"page {n}" for n in range(1, 14)}
    pages[2] = SOURCE
    result = {
        "target": "entities/mosaic",
        "verdict": "supported",
        "evidence": [{"page": 2, "quote": SOURCE}],
    }
    body = {"targets": [result]}
    if damage == "wrong_page":
        result["evidence"][0]["page"] = 11
    if damage == "footer_page":
        result["evidence"][0]["page"] = 17
    if damage == "wrong_quote":
        result["evidence"][0]["quote"] = "Mosaic owns IAM"
    if damage == "missing":
        body["targets"] = []
    if damage == "extra":
        result["target"] = "entities/other"
    if damage == "duplicate":
        body["targets"].append(result)
    if damage == "unclear":
        result["verdict"] = "unclear"
    if damage == "scalar":
        body = 1
    if damage == "extra_field":
        result["confidence"] = 1
    raw = json.dumps(body)
    if damage == "fenced":
        raw = "```json\n" + raw + "\n```"
    with pytest.raises(GroundingError):
        validate_audit(raw, {"entities/mosaic": "claim"}, pages)


@pytest.mark.asyncio
async def test_oversized_batch_fails_closed_without_call_or_dropped_coverage(monkeypatch):
    from openkb import grounding

    monkeypatch.setattr(grounding, "MAX_AUDIT_CHARS", 100)

    async def forbidden(*args, **kwargs):
        pytest.fail("oversized audit must not be sent")

    with pytest.raises(grounding.GroundingError, match="bound"):
        await grounding.audit_candidates(
            {1: SOURCE}, {"entities/mosaic": "Mosaic owns IAM"}, "mock/model", forbidden
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("long_doc", [False, True])
async def test_original_pages_reach_generation_and_compilation_reasoning(
    tmp_path, monkeypatch, long_doc
):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir()
    original = "Original: Vesper reviews code; unrelated system owns IAM."
    (wiki / "sources/report.json").write_text(json.dumps([{"page": 2, "content": original}]))
    summary = wiki / "summaries/report.md"
    summary.write_text("wrong summary: Vesper owns IAM")
    source = tmp_path / "source.txt"
    source.write_text(original)
    monkeypatch.setattr(
        "openkb.config.resolve_effective_config",
        lambda kb: ({"compilation_reasoning_effort": "medium"}, None),
    )
    seen = []

    def call(model, messages, step, **kwargs):
        assert original in str(messages)
        assert "physical" in str(messages)
        assert kwargs["bundle"].reasoning_effort == "medium"
        seen.append(step)
        return json.dumps({"content": "Vesper reviews code"})

    async def compile_pages(*args, **kwargs):
        assert original in str(args[4])
        assert kwargs["source_pages"] == {2: original}
        assert kwargs["bundle"].reasoning_effort == "medium"

    monkeypatch.setattr(compiler, "_llm_call", call)
    monkeypatch.setattr(compiler, "_compile_concepts", compile_pages)
    if long_doc:
        await compiler.compile_long_doc("report", summary, "id", tmp_path, "mock/model")
    else:
        await compiler.compile_short_doc("report", source, tmp_path, "mock/model")
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_compiler_cancellation_drains_generation_before_return(tmp_path, monkeypatch):
    import asyncio

    wiki = tmp_path / "wiki"
    (wiki / "summaries").mkdir(parents=True)
    raw = tmp_path / "raw" / "report.pdf"
    raw.parent.mkdir()
    raw.write_bytes(b"original raw")
    cache = tmp_path / "pageindex" / "tree.json"
    cache.parent.mkdir()
    cache.write_bytes(b"cached index")
    started = asyncio.Event()
    cleaned = asyncio.Event()
    monkeypatch.setattr(
        compiler,
        "_llm_call",
        lambda *a, **k: json.dumps({"create": [{"name": "review", "title": "Review"}]}),
    )

    async def page(*a, **k):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.set()

    monkeypatch.setattr(compiler, "_llm_call_page_async", page)
    task = asyncio.create_task(
        compiler._compile_concepts(wiki, tmp_path, "mock/model", {}, {}, "summary", "report", 2)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set(), "rollback must not race still-running generation cleanup"
    assert raw.read_bytes() == b"original raw"
    assert cache.read_bytes() == b"cached index"
    assert not list(wiki.rglob("*.md"))


@pytest.mark.asyncio
async def test_ambiguous_candidates_without_original_pages_require_manual_review():
    from openkb.grounding import GroundingError, audit_candidates

    async def forbidden(*a, **k):
        pytest.fail("summary must never be passed off as original source")

    with pytest.raises(GroundingError, match="source"):
        await audit_candidates(
            {}, {"summaries/report": "Prism owns review, search, and IAM."}, "mock/model", forbidden
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("long_doc", [False, True])
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_combined_product_publication_has_same_page_provenance(
    tmp_path, monkeypatch, long_doc
):
    wiki = tmp_path / "wiki"
    for folder in ("sources", "summaries"):
        (wiki / folder).mkdir(parents=True)
    text = "Vesper provides code review, code search, and access management."
    (wiki / "sources/report.json").write_text(
        json.dumps([{"page": 1, "content": "Unrelated first page"}, {"page": 2, "content": text}])
    )
    summary = wiki / "summaries/report.md"
    summary.write_text(text)
    source = tmp_path / "source.txt"
    source.write_text(text)
    calls = []

    def sync(model, messages, step, **kwargs):
        if step == "concepts-plan":
            return json.dumps(
                {"entities": {"create": [{"name": "vesper", "title": "Vesper", "type": "product"}]}}
            )
        if step in ("summary-rewrite", "overview"):
            return text
        return json.dumps({"content": text})

    async def call(model, messages, step, **kwargs):
        if step == "source-attribution-audit":
            calls.append(step)
            payload = json.loads(messages[1]["content"])
            return json.dumps(
                {
                    "targets": [
                        {
                            "target": key,
                            "verdict": "supported",
                            "evidence": [{"doc": "report", "page": 2, "quote": text}],
                        }
                        for key in payload["candidates"]
                    ]
                }
            )
        return json.dumps(
            {
                "description": "Vesper combines all three functions",
                "content": text,
                "type": "product",
            }
        )

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", call)
    if long_doc:
        await compiler.compile_long_doc("report", summary, "id", tmp_path, "mock/model")
    else:
        await compiler.compile_short_doc("report", source, tmp_path, "mock/model")
    assert calls == ["source-attribution-audit"]
    assert "[source report p. 2]" in (wiki / "entities/vesper.md").read_text()
    assert "[source report p. 2]" in summary.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.usefixtures("enable_grounding_audit")
async def test_summary_description_is_audited_before_publication(tmp_path, monkeypatch, planned):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    (tmp_path / "wiki/summaries").mkdir(parents=True)

    def sync(model, messages, step, **kwargs):
        if step == "concepts-plan":
            return json.dumps({"create": [{"name": "review"}]} if planned else {})
        if step == "summary-rewrite":
            return "General review workflow."
        return json.dumps({"description": "Mosaic owns IAM", "content": "General review workflow."})

    async def call(model, messages, step, **kwargs):
        if step == "source-attribution-audit":
            payload = json.loads(messages[1]["content"])
            assert "Mosaic owns IAM" in payload["candidates"]["summaries/report"]
            return json.dumps(
                {
                    "targets": [
                        {"target": "summaries/report", "verdict": "unsupported", "evidence": []}
                    ]
                }
            )
        return json.dumps({"content": "General review workflow."})

    monkeypatch.setattr(compiler, "_llm_call", sync)
    monkeypatch.setattr(compiler, "_llm_call_async", call)
    with pytest.raises(RuntimeError, match="Grounding"):
        await compiler.compile_short_doc("report", source, tmp_path, "mock/model")
    assert not list((tmp_path / "wiki").rglob("*.md"))
