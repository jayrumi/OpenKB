"""Loop ownership, not document completion, controls LLM client shutdown."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
async def test_cleanup_closes_only_current_loop_entries_and_evicts_them(monkeypatch):
    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache

    from openkb.llm_lifecycle import close_current_loop_clients

    cache = LLMClientCache()
    own = SimpleNamespace(aclose=AsyncMock())
    foreign = SimpleNamespace(aclose=AsyncMock())
    sync = SimpleNamespace(close=Mock())
    loop_id = id(asyncio.get_running_loop())
    own_key = f"async_httpx_client-chatgpt-{loop_id}"
    sync_key = f"httpx_client-{loop_id}"
    other_key = "async_httpx_client-other-foreign-loop"
    cache.cache_dict[own_key] = SimpleNamespace(client=own)
    cache.cache_dict[sync_key] = SimpleNamespace(client=sync)
    cache.cache_dict[other_key] = SimpleNamespace(client=foreign)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", cache)
    global_close = AsyncMock(side_effect=AssertionError("No global cleanup"))
    monkeypatch.setattr(litellm, "close_litellm_async_clients", global_close)

    await close_current_loop_clients()
    own.aclose.assert_awaited_once()
    sync.close.assert_called_once()
    foreign.aclose.assert_not_awaited()
    assert own_key not in cache.cache_dict and sync_key not in cache.cache_dict
    assert other_key in cache.cache_dict
    global_close.assert_not_awaited()


def test_owned_loop_cleans_up_even_when_coroutine_fails(monkeypatch):
    from openkb import llm_lifecycle

    cleanup = AsyncMock()
    monkeypatch.setattr(llm_lifecycle, "close_current_loop_clients", cleanup)

    async def fail():
        raise ValueError("expected failure")

    with pytest.raises(ValueError, match="expected failure"):
        llm_lifecycle.run_owned_llm_loop(fail())
    cleanup.assert_awaited_once()


def test_api_cleans_up_only_at_lifespan_shutdown(monkeypatch):
    from fastapi.testclient import TestClient

    from openkb import llm_lifecycle
    from openkb.api import create_app

    cleanup = AsyncMock()
    monkeypatch.setattr(llm_lifecycle, "close_current_loop_clients", cleanup)
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/meta").status_code == 200
        cleanup.assert_not_awaited()
    cleanup.assert_awaited_once()


def test_owned_loop_rejected_startup_does_not_leak_coroutines(monkeypatch):
    from openkb import llm_lifecycle

    captured = []

    def reject(coroutine):
        captured.append(coroutine)
        raise RuntimeError("loop startup rejected")

    monkeypatch.setattr(llm_lifecycle.asyncio, "run", reject)

    async def work():
        return None

    task = work()
    try:
        with pytest.raises(RuntimeError, match="startup rejected"):
            llm_lifecycle.run_owned_llm_loop(task)
        assert task.cr_frame is None
        assert captured[0].cr_frame is None
    finally:
        task.close()
        for coroutine in captured:
            coroutine.close()
