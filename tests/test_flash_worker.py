import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_policy_bounds_summary_and_sync_description_calls(monkeypatch):
    import pageindex

    from openkb.flash_worker import completion_policy
    from openkb.indexing_policy import FlashAbort

    fake_utils = SimpleNamespace(
        _acompletion=object(),
        _completion=object(),
        _claude_none_kwargs=lambda model, kw: kw,
        _add_usage=lambda response: None,
    )
    monkeypatch.setitem(sys.modules, "pageindex.utils", fake_utils)
    monkeypatch.setattr(pageindex, "utils", fake_utils, raising=False)
    calls = []

    async def provider(**kw):
        calls.append(kw)
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=10),
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=provider))
    with pytest.raises(FlashAbort, match="budget"):
        with completion_policy({"max_calls": 1}):
            assert asyncio.run(fake_utils._acompletion("exact", [], {})) == "summary"
            with pytest.raises(FlashAbort, match="budget"):
                fake_utils._completion("exact", [], {}, False)
    assert len(calls) == 1


def test_quota_cancels_inflight_sibling_requests(monkeypatch):
    import pageindex

    from openkb.flash_worker import completion_policy
    from openkb.indexing_policy import FlashAbort

    fake_utils = SimpleNamespace(
        _acompletion=object(),
        _completion=object(),
        _claude_none_kwargs=lambda model, kw: kw,
        _add_usage=lambda response: None,
    )
    monkeypatch.setitem(sys.modules, "pageindex.utils", fake_utils)
    monkeypatch.setattr(pageindex, "utils", fake_utils, raising=False)
    entered = []

    async def provider(**kw):
        entered.append(kw)
        if len(entered) == 1:
            await asyncio.sleep(0.01)
            raise RuntimeError("usage_limit_reached")
        await asyncio.Event().wait()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=provider))

    async def summarize():
        with completion_policy({}):
            await asyncio.wait_for(
                asyncio.gather(
                    fake_utils._acompletion("exact", [], {}),
                    fake_utils._acompletion("exact", [], {}),
                    return_exceptions=True,
                ),
                timeout=0.2,
            )

    with pytest.raises(FlashAbort, match="quota"):
        asyncio.run(summarize())
    assert len(entered) == 2


def test_quota_latch_blocks_further_nodes_even_if_sdk_swallows_abort(monkeypatch):
    import pageindex

    from openkb.flash_worker import completion_policy
    from openkb.indexing_policy import FlashAbort

    fake_utils = SimpleNamespace(
        _acompletion=object(),
        _completion=object(),
        _claude_none_kwargs=lambda model, kw: kw,
        _add_usage=lambda response: None,
    )
    monkeypatch.setitem(sys.modules, "pageindex.utils", fake_utils)
    monkeypatch.setattr(pageindex, "utils", fake_utils, raising=False)
    calls = []

    async def provider(**kw):
        calls.append(kw)
        raise RuntimeError("usage_limit_reached")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=provider))
    with pytest.raises(FlashAbort, match="quota"):
        with completion_policy({}):
            for _ in range(4):
                try:
                    asyncio.run(fake_utils._acompletion("exact", [], {}))
                except FlashAbort:
                    pass  # Simulates SDK gather(return_exceptions=True).
    assert len(calls) == 1


def test_worker_uses_actual_sdk_api_and_preserves_exact_model(tmp_path, monkeypatch):
    from openkb import flash_worker as worker

    monkeypatch.setattr(worker, "version", lambda name: "0.7.4")
    sdk = Mock()
    sdk.submit_document.return_value = {"doc_id": "pi-test", "usage": {"total_tokens": 3}}
    sdk.get_document.return_value = {
        "name": "book.pdf",
        "description": "desc",
        "status": "completed",
        "pageNum": 1,
    }
    sdk.get_tree.return_value = {"status": "completed", "result": [{"node_id": "0000"}]}
    factory = Mock(return_value=sdk)
    result = worker.execute(
        {
            "pdf_path": str(tmp_path / "book.pdf"),
            "model": "chatgpt/gpt-6-luna",
            "summary_model": "chatgpt/gpt-6-luna",
        },
        client_factory=factory,
    )
    assert result["structure"] == [{"node_id": "0000"}]
    assert factory.call_args.kwargs["mode"] == "local"
    assert factory.call_args.kwargs["index_model"] == "chatgpt/gpt-6-luna"
    assert factory.call_args.kwargs["summary_model"] == "chatgpt/gpt-6-luna"
    assert factory.call_args.kwargs["summary_concurrency"] == 1
    assert factory.call_args.kwargs["optimize"] == "off"
    sdk.submit_document.assert_called_once_with(str(tmp_path / "book.pdf"), mode="flash")
    sdk.get_tree.assert_called_once_with("pi-test", node_summary=True, include_text=True)


def test_worker_rejects_unbounded_non_litellm_backend(monkeypatch):
    from openkb import flash_worker as worker

    monkeypatch.setattr(worker, "version", lambda name: "0.7.4")
    factory = Mock()
    with pytest.raises(ValueError, match="bounded"):
        worker.execute(
            {"pdf_path": "book.pdf", "model": "claude-code/sonnet"}, client_factory=factory
        )
    factory.assert_not_called()


def test_worker_rejects_incomplete_sdk_result(tmp_path, monkeypatch):
    from openkb import flash_worker as worker

    monkeypatch.setattr(worker, "version", lambda name: "0.7.4")
    sdk = Mock()
    sdk.submit_document.return_value = {"doc_id": "pi-test", "usage": {}}
    sdk.get_document.return_value = {"status": "processing"}
    with pytest.raises(ValueError, match="completed"):
        worker.execute(
            {"pdf_path": str(tmp_path / "book.pdf"), "model": "exact"},
            client_factory=lambda **kw: sdk,
        )
