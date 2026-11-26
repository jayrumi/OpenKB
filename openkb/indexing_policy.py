"""Bounded per-request policy, used only in the isolated Flash process."""

from __future__ import annotations

import asyncio
import importlib
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from litellm.exceptions import Timeout

_reasoning = importlib.import_module("openkb.reasoning" if __package__ else "reasoning")
_fallback_log = _reasoning._fallback_log
reasoning_fallback_allowed = _reasoning.reasoning_fallback_allowed


class FlashAbort(BaseException):
    """Stop the complete indexing run; SDK retry/fallback must not swallow this."""


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def quota_error(error: BaseException) -> bool:
    return any(
        any(
            marker in str(item).lower()
            for marker in (
                "usage_limit_reached",
                "insufficient_quota",
                "quota_exceeded",
                "billing_hard_limit_reached",
                "exceeded your current quota",
            )
        )
        for item in _chain(error)
    )


def retry_delay(error, attempt: int, backoff: float = 2) -> float:
    headers = getattr(error, "litellm_response_headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    try:
        delay = float(value) if value is not None else 0
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value))
            delay = (parsed - datetime.now(timezone.utc)).total_seconds() if parsed else 0
        except (TypeError, ValueError, OverflowError):
            delay = 0
    if not math.isfinite(delay):
        delay = 0
    # Never retry earlier than Retry-After; an excessive delay aborts instead.
    if delay > 60:
        raise FlashAbort("Retry-After exceeds bounded indexing wait")
    return max(0, delay, min(60, backoff * 2**attempt))


async def bounded_completion(
    completion,
    model,
    messages,
    sent,
    *,
    sleep=asyncio.sleep,
    timeout=120,
    retries=2,
    backoff=2,
    on_reasoning_fallback=None,
):
    """Bounded attempts for a single request, never an entire document."""
    sent = dict(sent)
    attempt = 0
    fallback_used = False
    while attempt <= retries:
        try:
            return await completion(
                **{
                    **sent,
                    "model": model,
                    "messages": messages,
                    "max_retries": 0,
                    "timeout": timeout,
                }
            )
        except Exception as error:
            if not fallback_used and reasoning_fallback_allowed(
                error, model, sent.get("reasoning_effort"), stream=sent.get("stream", False)
            ):
                fallback_used = True
                sent["reasoning_effort"] = "low"
                _fallback_log()
                if on_reasoning_fallback is not None:
                    on_reasoning_fallback("none", "low")
                continue
            if quota_error(error):
                raise FlashAbort("Flash indexing stopped: quota exhausted") from None
            status = getattr(error, "status_code", None)
            if status in (401, 403) or any(
                word in type(error).__name__.lower() for word in ("auth", "permission")
            ):
                raise FlashAbort("Flash indexing auth is unsupported or unavailable") from None
            transient = status in (429, 500, 502, 503, 504) or isinstance(
                error, (TimeoutError, ConnectionError, Timeout)
            )
            if not transient or attempt == retries:
                raise FlashAbort("Flash LLM request failed; no document retry") from None
            await sleep(retry_delay(error, attempt, backoff))
            attempt += 1
