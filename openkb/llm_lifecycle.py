"""Own LLM clients at event-loop boundaries, not individual documents.

The pinned LiteLLM cache adds '-<loop id>' to keys. Its global shutdown closes
clients from unrelated loops and retains stale closed entries, so never call
that shutdown from a document coroutine or a worker thread in a live server.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def close_current_loop_clients() -> None:
    """Best-effort cleanup for the loop that the caller is about to shut down.

    Only handles bearing the pinned SDK's current-loop cache suffix are removed
    and closed. Live clients for other loops and process-global handlers remain
    untouched; one CLI thread must not close a server's active query client.
    """
    import litellm

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        return
    entries = getattr(cache, "cache_dict", {})
    suffix = f"-{id(asyncio.get_running_loop())}"
    for key, handler in list(entries.items()):
        if not isinstance(key, str) or not key.endswith(suffix):
            continue
        if entries.get(key) is not handler:
            continue
        try:
            # delete_cache accepts the already-qualified cache key; do not use
            # get_cache/set_cache here (they append the loop suffix again).
            cache.delete_cache(key)
            client = getattr(handler, "client", handler)
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            logger.debug("Loop-owned LLM cleanup failed: %s", type(exc).__name__)


def run_owned_llm_loop(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a finite CLI/thread job and close its clients before its loop exits."""

    async def run() -> T:
        try:
            return await coroutine
        finally:
            await close_current_loop_clients()

    owned = run()
    try:
        return asyncio.run(owned)
    finally:
        # asyncio.run may reject startup (e.g. an already-running loop). Do not
        # leave either coroutine unclosed when no loop ever awaited it.
        owned.close()
        coroutine.close()
