"""Actual add coordinator + Flash bridge rollback/reuse, model boundary mocked."""

import json
from unittest.mock import AsyncMock, patch

import pymupdf
import yaml

from openkb.agent.tools import get_wiki_page_content
from openkb.cli import add_single_file
from openkb.flash_adapter import SDK_VERSION


def test_flash_compilation_failure_rolls_back_but_reuses_successful_index(kb_dir, tmp_path):
    pdf = tmp_path / "warehouse.pdf"
    with pymupdf.open() as document:
        for text in (
            "Grain is one order line item.",
            "Type 2 uses valid_from and valid_to.",
            "HTTP 429 hard quota aborts indexing.",
        ):
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(pdf)
    cache = tmp_path / "index-cache"
    (kb_dir / ".openkb/config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "chatgpt/gpt-6.2-sol",
                "indexing_model": "chatgpt/gpt-6-luna",
                "compilation_model": "chatgpt/gpt-6.2-sol",
                "indexing_backend": "flash",
                "flash_python": "/usr/bin/python3",
                "flash_cache_dir": str(cache),
                "pageindex_threshold": 1,
            }
        )
    )
    previous = kb_dir / "wiki/summaries/existing.md"
    previous.write_text("existing document must survive")
    result = {
        "sdk_version": SDK_VERSION,
        "doc_name": "warehouse.pdf",
        "doc_description": "Warehouse modeling notes",
        "page_count": 3,
        "structure": [
            {
                "title": f"Section {i}",
                "node_id": f"000{i}",
                "start_index": i,
                "end_index": i,
                "summary": text,
            }
            for i, text in enumerate(("Grain", "History", "Quotas"), 1)
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    with (
        patch("openkb.cli._setup_llm_key"),
        patch("openkb.cli.time.sleep"),
        patch("openkb.flash_adapter.run_worker", return_value=result) as worker,
        patch("openkb.agent.compiler.compile_long_doc", new_callable=AsyncMock) as compile_doc,
    ):
        compile_doc.side_effect = RuntimeError("mock compilation failure")
        assert add_single_file(pdf, kb_dir) == "failed"
        assert previous.read_text() == "existing document must survive"
        assert not list((kb_dir / ".openkb/files/default").glob("*.pdf"))
        assert list(cache.glob("*.json")), "Completed expensive index must survive rollback"
        compile_doc.side_effect = None
        compile_doc.return_value = None
        assert add_single_file(pdf, kb_dir) == "added"
        assert worker.call_count == 1, "Retry compilation must not reindex the PDF"
        assert compile_doc.call_args.args[4] == "chatgpt/gpt-6.2-sol"
    sources = list((kb_dir / "wiki/sources").glob("*.json"))
    assert len(sources) == 1
    pages = json.loads(sources[0].read_text())
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "valid_from" in get_wiki_page_content(sources[0].stem, "2", str(kb_dir / "wiki"))
    assert previous.read_text() == "existing document must survive"
