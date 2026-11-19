from unittest.mock import MagicMock

from openkb.indexer import index_long_document


def test_legacy_chatgpt_indexing_uses_oauth_validator_not_api_key(kb_dir, tmp_path, monkeypatch):
    model = "chatgpt/gpt-6-luna"
    monkeypatch.delenv("PAGEINDEX_API_KEY", raising=False)
    monkeypatch.setattr("openkb.indexer.resolve_effective_config", lambda p: ({"model": model}, {}))
    client = MagicMock()
    client.collection.return_value.add.return_value = "test-doc"
    client.collection.return_value.get_document.return_value = {
        "doc_name": "sample",
        "doc_description": "test",
        "structure": [],
    }
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("openkb.provider_compat.OAuthPageIndexClient", factory)
    monkeypatch.setattr(
        "openkb.indexer._convert_pdf_to_pages",
        lambda *a: [{"page": 1, "content": "local text", "images": []}],
    )
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = index_long_document(pdf, kb_dir)
    assert result.doc_id == "test-doc"
    assert factory.call_args.kwargs["model"] == model
    assert factory.call_args.kwargs["api_key"] is None
