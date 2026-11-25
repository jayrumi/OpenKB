"""Offline regressions for required-page failures and index-free recovery."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from openkb.agent import compiler


def _mock_pipeline(monkeypatch, *, failed: str | None = "concept: fact-table-grain"):
    plan = {
        "create": [{"name": "data-modeling"}, {"name": "fact-table-grain"}],
        "update": [],
        "related": [],
        "entities": {"create": [{"name": "warehouse", "type": "other"}]},
    }

    def sync_call(model, messages, label, **kwargs):
        if label == "summary":
            return json.dumps({"content": "Summary"})
        if label == "overview":
            return "Overview"
        if "plan" in label or "concept" in label:
            return json.dumps(plan)
        return "Summary linking [[concepts/fact-table-grain]]"

    async def page_call(model, messages, label, **kwargs):
        if label == failed:
            raise RuntimeError("provider response missing usage")
        return json.dumps({"content": "Body linking [[concepts/fact-table-grain]]"})

    monkeypatch.setattr(compiler, "_llm_call", sync_call)
    page = AsyncMock(side_effect=page_call)
    monkeypatch.setattr(compiler, "_llm_call_page_async", page)
    monkeypatch.setattr(compiler, "_close_async_llm_clients", AsyncMock())
    return page


@pytest.mark.parametrize("kind", ["short", "long"])
@pytest.mark.parametrize("failed", ["concept: fact-table-grain", "entity: warehouse"])
def test_required_page_failure_never_publishes_partial_success(kb_dir, monkeypatch, kind, failed):
    wiki = kb_dir / "wiki"
    source = wiki / "sources" / "warehouse.md"
    source.write_text("Warehouse modeling")
    (wiki / "sources" / "warehouse.json").write_text(
        json.dumps([{"page": 1, "content": "Warehouse modeling"}])
    )
    summary = wiki / "summaries" / "warehouse.md"
    summary.write_text("Existing indexed summary")
    index = wiki / "index.md"
    index.write_text("# Index\n\n## Documents\n\n## Concepts\n")
    before_index = index.read_bytes()
    _mock_pipeline(monkeypatch, failed=failed)

    with pytest.raises(RuntimeError, match="Incomplete compilation.*warehouse") as exc:
        if kind == "long":
            asyncio.run(compiler.compile_long_doc("warehouse", summary, "doc-id", kb_dir, "mock"))
        else:
            asyncio.run(compiler.compile_short_doc("warehouse", source, kb_dir, "mock"))

    assert failed in str(exc.value)
    assert "provider response missing usage" not in str(exc.value)
    assert str(exc.value.__cause__) == "provider response missing usage"
    assert index.read_bytes() == before_index
    assert list((wiki / "concepts").glob("*.md")) == []
    assert not (wiki / "entities" / "warehouse.md").exists()


def _seed_recovery(kb_dir):
    wiki = kb_dir / "wiki"
    (kb_dir / ".openkb" / "hashes.json").write_text(
        json.dumps(
            {
                "hash1": {
                    "name": "warehouse.pdf",
                    "doc_name": "warehouse",
                    "type": "long_pdf",
                    "doc_id": "indexed-id",
                },
                "hash2": {
                    "name": "other.pdf",
                    "doc_name": "other",
                    "type": "long_pdf",
                    "doc_id": "other-id",
                },
            }
        )
    )
    (wiki / "summaries" / "warehouse.md").write_text(
        "---\nsources: [raw/warehouse.pdf]\n---\nIndexed summary"
    )
    (wiki / "summaries" / "other.md").write_text("Other book, unchanged")
    (wiki / "concepts" / "data-modeling.md").write_text("Old body [[concepts/fact-table-grain]]")
    (wiki / "index.md").write_text("# Index\n\n## Documents\n\n## Concepts\n\n## Entities\n")
    (wiki / "log.md").write_text("# Log\n")
    cache = kb_dir / ".openkb" / "files" / "cached-index"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"completed index cache")
    from pageindex.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(str(kb_dir / ".openkb" / "pageindex.db"))
    storage.get_or_create_collection("default")
    for name, doc_id in (("warehouse", "indexed-id"), ("other", "other-id")):
        (wiki / "sources" / f"{name}.json").write_text(
            json.dumps([{"page": 1, "content": name}])
        )
        raw = kb_dir / "raw" / f"{name}.pdf"
        raw.write_bytes(f"%PDF-preservation-sentinel:{name}".encode())
        storage.save_document(
            "default",
            doc_id,
            {
                "doc_name": name,
                "doc_type": "pdf",
                "file_path": str(raw),
                "structure": [],
                "pages": [{"page": 1, "content": name}],
            },
        )
    storage.close()
    return wiki


@pytest.mark.parametrize("interface", ["cli", "rest"])
@pytest.mark.parametrize("kind", ["short", "long"])
def test_targeted_recompile_rolls_back_failure_without_touching_index(
    kb_dir, monkeypatch, interface, kind
):
    from click.testing import CliRunner

    from openkb.cli import cli, iter_recompile
    from openkb.locks import atomic_write_text

    wiki = _seed_recovery(kb_dir)
    if kind == "short":
        registry = kb_dir / ".openkb" / "hashes.json"
        entries = json.loads(registry.read_text())
        entries["hash1"]["type"] = "md"
        registry.write_text(json.dumps(entries))
        (wiki / "sources" / "warehouse.md").write_text("Source")
    before = {str(p.relative_to(kb_dir)): p.read_bytes() for p in kb_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *a: None)
    indexer = AsyncMock(side_effect=AssertionError("must not reindex"))
    monkeypatch.setattr("openkb.indexer.index_long_document", indexer)

    async def fail(*args, **kwargs):
        assert args[0] == "warehouse"
        atomic_write_text(wiki / "summaries" / "warehouse.md", "Damaged summary")
        atomic_write_text(wiki / "concepts" / "data-modeling.md", "Damaged body")
        atomic_write_text(wiki / "concepts" / "new.md", "Partial page")
        raise RuntimeError("incomplete generation")

    compile_mock = AsyncMock(side_effect=fail)
    monkeypatch.setattr(compiler, f"compile_{kind}_doc", compile_mock)
    if interface == "cli":
        result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "recompile", "warehouse"])
        assert "recompiled 0" in result.output
        assert "[OK]" not in result.output
    else:

        async def collect():
            return [event async for event in iter_recompile(kb_dir, doc_name="warehouse")]

        events = asyncio.run(collect())
        assert events[-1]["recompiled"] == 0
        assert events[-1]["docs"][0]["status"] == "error"
    compile_mock.assert_awaited_once()
    indexer.assert_not_called()
    for path, body in before.items():
        if path != "wiki/log.md":
            assert (kb_dir / path).read_bytes() == body, path
    assert not (wiki / "concepts" / "new.md").exists()


def test_real_add_failure_reports_zero_added_and_retry_repairs(kb_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from openkb.api import create_app
    from openkb.cli import add_single_file
    from openkb.state import HashRegistry

    wiki = _seed_recovery(kb_dir)
    before = {str(p.relative_to(kb_dir)): p.read_bytes() for p in kb_dir.rglob("*") if p.is_file()}
    monkeypatch.delenv("OPENKB_API_TOKEN", raising=False)
    monkeypatch.setattr("openkb.api_helpers.resolve_kb_alias", lambda _: kb_dir)
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *a: None)
    monkeypatch.setattr("openkb.cli.time.sleep", lambda *a: None)
    page = _mock_pipeline(monkeypatch)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/add",
        data={"kb": "test", "stream": "false"},
        files=[("files", ("new-notes.md", b"# Warehouse modeling", "text/markdown"))],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["added_count"] == 0
    assert payload["failed_count"] == 1
    assert payload["files"][0]["status"] == "failed"
    assert page.await_count == 6  # retry drains all three required tasks twice
    for path, body in before.items():
        assert (kb_dir / path).read_bytes() == body, path
    assert not (wiki / "concepts" / "fact-table-grain.md").exists()

    _mock_pipeline(monkeypatch, failed=None)
    assert add_single_file(kb_dir / "raw" / "new-notes.md", kb_dir) == "added"
    assert (wiki / "concepts" / "fact-table-grain.md").exists()
    assert len(HashRegistry(kb_dir / ".openkb" / "hashes.json").all_entries()) == 3


@pytest.mark.parametrize("interface", ["cli", "rest"])
def test_real_targeted_recompile_repairs_missing_page_without_reindex(
    kb_dir, monkeypatch, interface
):
    from click.testing import CliRunner

    from openkb.cli import cli, iter_recompile

    wiki = _seed_recovery(kb_dir)
    other = (wiki / "summaries" / "other.md").read_bytes()
    cache = (kb_dir / ".openkb" / "files" / "cached-index").read_bytes()
    registry = (kb_dir / ".openkb" / "hashes.json").read_bytes()
    preserved = {
        p: p.read_bytes()
        for p in [kb_dir / ".openkb" / "pageindex.db", *sorted((kb_dir / "raw").glob("*.pdf"))]
    }
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *a: None)
    indexer = AsyncMock(side_effect=AssertionError("must not reindex"))
    monkeypatch.setattr("openkb.indexer.index_long_document", indexer)
    page = _mock_pipeline(monkeypatch, failed=None)
    if interface == "cli":
        result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "recompile", "warehouse"])
        assert result.exit_code == 0, result.output
        assert "recompiled 1" in result.output
    else:

        async def collect():
            return [event async for event in iter_recompile(kb_dir, doc_name="warehouse")]

        events = asyncio.run(collect())
        assert events[-1]["recompiled"] == 1
        assert events[-1]["total"] == 1
    assert page.await_count == 3
    assert (wiki / "concepts" / "fact-table-grain.md").exists()
    assert "[[concepts/fact-table-grain]]" in (wiki / "concepts" / "data-modeling.md").read_text()
    assert (wiki / "summaries" / "other.md").read_bytes() == other
    assert (kb_dir / ".openkb" / "files" / "cached-index").read_bytes() == cache
    assert (kb_dir / ".openkb" / "hashes.json").read_bytes() == registry
    assert all(p.read_bytes() == content for p, content in preserved.items())
    indexer.assert_not_called()


@pytest.mark.parametrize("interface", ["cli", "rest"])
def test_dirty_recompile_rollback_stops_batch(kb_dir, monkeypatch, interface):
    from click.testing import CliRunner

    from openkb.add_coordinator import DirtyRollbackError
    from openkb.cli import cli, iter_recompile

    _seed_recovery(kb_dir)
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *a: None)
    compile_mock = AsyncMock(side_effect=RuntimeError("generation failed"))
    monkeypatch.setattr(compiler, "compile_long_doc", compile_mock)
    monkeypatch.setattr(
        "openkb.mutation.MutationSnapshot.rollback_best_effort",
        lambda _: RuntimeError("disk failure"),
    )
    if interface == "cli":
        result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "recompile", "--all", "--yes"])
        assert isinstance(result.exception, DirtyRollbackError)
        assert "recompiled 1" not in result.output
    else:

        async def collect():
            return [event async for event in iter_recompile(kb_dir, all_docs=True)]

        with pytest.raises(DirtyRollbackError):
            asyncio.run(collect())
    compile_mock.assert_awaited_once()


def test_cancelled_required_page_propagates_without_publishing(kb_dir, monkeypatch):
    wiki = _seed_recovery(kb_dir)
    before = (wiki / "index.md").read_bytes()
    page = _mock_pipeline(monkeypatch)
    page.side_effect = asyncio.CancelledError("cancelled")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            compiler.compile_long_doc(
                "warehouse", wiki / "summaries" / "warehouse.md", "indexed-id", kb_dir, "mock"
            )
        )
    assert (wiki / "index.md").read_bytes() == before
    assert not (wiki / "concepts" / "fact-table-grain.md").exists()


def test_recompile_cancellation_restores_pages_and_stops_batch(kb_dir, monkeypatch):
    from openkb.cli import iter_recompile
    from openkb.locks import atomic_write_text

    wiki = _seed_recovery(kb_dir)
    before = {p: p.read_bytes() for p in kb_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr("openkb.cli._setup_llm_key", lambda *a: None)

    async def cancel(name, *args, **kwargs):
        atomic_write_text(wiki / "summaries" / f"{name}.md", "partial summary")
        atomic_write_text(wiki / "concepts" / "data-modeling.md", "partial shared page")
        atomic_write_text(wiki / "concepts" / "cancelled-new.md", "must roll back")
        raise asyncio.CancelledError()

    compile_mock = AsyncMock(side_effect=cancel)
    monkeypatch.setattr(compiler, "compile_long_doc", compile_mock)

    async def collect():
        return [event async for event in iter_recompile(kb_dir, all_docs=True)]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect())
    compile_mock.assert_awaited_once()
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (wiki / "concepts" / "cancelled-new.md").exists()
