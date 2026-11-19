from unittest.mock import Mock

import pytest
from pageindex import PageIndexClient
from test_flash_adapter import result

from openkb.indexer import index_long_document


def test_flash_preserves_artifacts_and_legacy_retrieval(tmp_path, monkeypatch):
    from openkb import flash_adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"PDF")
    config = {
        "indexing_backend": "flash",
        "flash_python": "/usr/bin/python3",
        "indexing_model": "chatgpt/gpt-6-luna",
    }
    monkeypatch.setattr("openkb.indexer.resolve_effective_config", lambda kb: (config, {}))
    run = Mock(return_value=result())
    monkeypatch.setattr(flash_adapter, "index_pdf", run)
    pages = [{"page": 1, "content": "page text", "images": [{"path": "images/safe/figure.png"}]}]
    extraction = Mock(return_value=pages)
    monkeypatch.setattr("openkb.indexer._convert_pdf_to_pages", extraction)
    indexed = index_long_document(pdf, tmp_path, "safe")
    assert indexed.description == "summary"
    assert (tmp_path / "wiki/summaries/safe.md").exists()
    assert "figure.png" in (tmp_path / "wiki/sources/safe.json").read_text()
    assert extraction.call_args.args == (pdf, "safe", tmp_path / "wiki/sources/images/safe")
    col = PageIndexClient(model="ollama/test", storage_path=str(tmp_path / ".openkb")).collection()
    assert (
        col.get_document(indexed.doc_id, include_text=True)["structure"][0]["text"] == "page text"
    )
    assert col.get_page_content(indexed.doc_id, "1")[0]["content"] == "page text"
    assert len(list((tmp_path / ".openkb/files/default").glob("*.pdf"))) == 1
    # Selecting Flash must not automatically reindex content already in old storage.
    assert index_long_document(pdf, tmp_path, "safe").doc_id == indexed.doc_id
    assert run.call_count == 1


def test_artifact_failure_keeps_cache_but_not_partial_legacy_index(tmp_path, monkeypatch):
    from pageindex.storage.sqlite import SQLiteStorage

    from openkb import flash_adapter, indexer

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"PDF")
    config = {
        "indexing_backend": "flash",
        "flash_python": "/usr/bin/python3",
        "model": "exact",
        "flash_cache_dir": str(tmp_path / "cache"),
    }
    monkeypatch.setattr(indexer, "resolve_effective_config", lambda kb: (config, {}))
    runner = Mock(return_value=result())
    monkeypatch.setattr(flash_adapter, "run_worker", runner)
    monkeypatch.setattr(
        indexer, "_convert_pdf_to_pages", lambda *a: [{"page": 1, "content": "body"}]
    )
    writer = indexer._write_long_doc_artifacts
    monkeypatch.setattr(
        indexer, "_write_long_doc_artifacts", Mock(side_effect=RuntimeError("disk"))
    )
    with pytest.raises(RuntimeError, match="disk"):
        index_long_document(pdf, tmp_path)
    storage = SQLiteStorage(str(tmp_path / ".openkb/pageindex.db"))
    assert storage.list_documents("default") == []
    storage.close()
    assert not list((tmp_path / ".openkb/files/default").glob("*.pdf"))
    monkeypatch.setattr(indexer, "_write_long_doc_artifacts", writer)
    assert index_long_document(pdf, tmp_path).description == "summary"
    assert runner.call_count == 1


def test_legacy_quota_never_retries_entire_document(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    client = MagicMock()
    client.collection.return_value.add.side_effect = RuntimeError("usage_limit_reached secret")
    monkeypatch.setattr("openkb.indexer.PageIndexClient", lambda **kw: client)
    monkeypatch.setattr("openkb.indexer.resolve_effective_config", lambda kb: ({}, {}))
    with pytest.raises(RuntimeError, match="quota"):
        index_long_document(tmp_path / "book.pdf", tmp_path)
    assert client.collection.return_value.add.call_count == 1


def test_unknown_backend_rejected_without_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openkb.indexer.resolve_effective_config", lambda kb: ({"indexing_backend": "typo"}, {})
    )
    with pytest.raises(ValueError, match="indexing_backend"):
        index_long_document(tmp_path / "book.pdf", tmp_path)
