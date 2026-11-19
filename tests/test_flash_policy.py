import asyncio
from types import SimpleNamespace

import pytest


def test_configured_request_policy_is_applied():
    from openkb.indexing_policy import FlashAbort, bounded_completion

    calls, waits = [], []

    async def sleep(delay):
        waits.append(delay)

    async def completion(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("temporary")
        error.status_code = 429
        raise error

    with pytest.raises(FlashAbort):
        asyncio.run(
            bounded_completion(
                completion, "exact", [], {}, sleep=sleep, timeout=7, retries=1, backoff=5
            )
        )
    assert len(calls) == 2 and waits == [5]
    assert all(call["timeout"] == 7 for call in calls)


def test_auth_failure_is_a_blocker_without_retry():
    from openkb.indexing_policy import FlashAbort, bounded_completion

    async def completion(**kwargs):
        error = RuntimeError("do not log this secret")
        error.status_code = 401
        raise error

    with pytest.raises(FlashAbort, match="auth"):
        asyncio.run(bounded_completion(completion, "chatgpt/exact", [], {}))


def test_retry_exhaustion_is_three_attempts():
    from openkb.indexing_policy import FlashAbort, bounded_completion

    calls = []

    async def sleep(delay):
        pass

    async def completion(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("temporary")
        error.status_code = 429
        raise error

    with pytest.raises(FlashAbort, match="request"):
        asyncio.run(bounded_completion(completion, "exact", [], {}, sleep=sleep))
    assert len(calls) == 3


def test_litellm_timeout_retries_twice_before_aborting():
    from litellm import Timeout

    from openkb.indexing_policy import FlashAbort, bounded_completion

    calls, waits = [], []

    async def sleep(delay):
        waits.append(delay)

    async def completion(**kwargs):
        calls.append(kwargs)
        raise Timeout(message="request timed out", model="exact", llm_provider="openai")

    with pytest.raises(FlashAbort, match="request"):
        asyncio.run(
            bounded_completion(completion, "exact", [], {}, sleep=sleep, retries=2)
        )
    assert len(calls) == 3
    assert waits == [2, 4]


def test_quota_aborts_without_retry():
    from openkb.indexing_policy import FlashAbort, bounded_completion

    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("usage_limit_reached private-token")

    with pytest.raises(FlashAbort, match="quota"):
        asyncio.run(bounded_completion(completion, "chatgpt/gpt-6-luna", [], {}, sleep=None))
    assert len(calls) == 1


def test_transient_retry_is_bounded_and_honors_retry_after():
    from openkb.indexing_policy import bounded_completion

    calls, waits = [], []

    async def sleep(seconds):
        waits.append(seconds)

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            error = RuntimeError("rate limited")
            error.status_code = 429
            error.response = SimpleNamespace(headers={"Retry-After": "3"})
            raise error
        return "ok"

    assert asyncio.run(bounded_completion(completion, "exact/model", [], {}, sleep=sleep)) == "ok"
    assert waits == [3, 4]
    assert all(
        c["model"] == "exact/model" and c["max_retries"] == 0 and c["timeout"] == 120 for c in calls
    )
