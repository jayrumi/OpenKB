"""Parent integration regressions for independently implemented v6 changes."""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest


def test_worker_enters_layout_policy_before_sdk_summary_submission(monkeypatch):
    import pageindex

    from openkb import flash_structure, flash_worker

    events = []

    @contextmanager
    def policy():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class Client:
        def __init__(self, **kwargs):
            pass

        def submit_document(self, path, mode):
            assert events == ["enter"]
            events.append("submit")
            return {"doc_id": "fixture", "usage": {}}

        def get_document(self, doc_id):
            return {"status": "completed", "name": "fixture", "pageNum": 1}

        def get_tree(self, *args, **kwargs):
            return {"status": "completed", "result": []}

    monkeypatch.setattr(flash_worker, "version", lambda package: "0.7.4")
    monkeypatch.setattr(flash_structure, "flash_structure_policy", policy)
    monkeypatch.setattr(pageindex, "PageIndexClient", Client)
    flash_worker.execute({"model": "chatgpt/gpt-6-luna", "pdf_path": "fixture.pdf"})
    assert events == ["enter", "submit", "exit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["short", "long"])
async def test_explicit_compilation_effort_preserves_cli_headers_and_timeout(
    tmp_path, monkeypatch, kind
):
    from openkb import config
    from openkb.agent import compiler

    monkeypatch.setattr(
        config,
        "resolve_effective_config",
        lambda kb: ({"compilation_reasoning_effort": "medium"}, {}),
    )
    monkeypatch.setattr(config, "get_extra_headers", lambda: {"X-Test-Header": "retained"})
    monkeypatch.setattr(config, "get_timeout", lambda: 37)
    (tmp_path / "wiki/sources").mkdir(parents=True)
    (tmp_path / "wiki/sources/doc.json").write_text(
        json.dumps([{"page": 1, "content": "A simple document."}])
    )
    source = tmp_path / "source.md"
    source.write_text("A simple document.")
    summary = tmp_path / "summary.md"
    summary.write_text("# Document\nA summary.")
    seen = []

    def call(*args, **kwargs):
        seen.append(kwargs["bundle"])
        return '{"description":"doc","content":"A summary."}'

    monkeypatch.setattr(compiler, "_llm_call", call)
    monkeypatch.setattr(compiler, "_compile_concepts", AsyncMock())
    if kind == "short":
        await compiler.compile_short_doc("doc", source, tmp_path, "chatgpt/gpt-6-sol")
    else:
        await compiler.compile_long_doc("doc", summary, "doc-id", tmp_path, "chatgpt/gpt-6-sol")
    assert len(seen) == 1
    assert seen[0].reasoning_effort == "medium"
    assert seen[0].extra_headers == {"X-Test-Header": "retained"}
    assert seen[0].timeout == 37
