"""Request-local reasoning selection and narrow compatibility policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def with_reasoning_effort(bundle, effort):
    """Preserve identity when unset; never resolve credentials or mutate input."""
    if effort is None:
        return bundle
    from openkb.config import LlmCredentialBundle

    return replace(bundle or LlmCredentialBundle(), reasoning_effort=effort)


def stage_reasoning_kwargs(config, stage, bundle=None) -> dict[str, Any]:
    """Optional bundle override for legacy CLI and ancillary stage entrypoints.

    Unset settings leave old call signatures unchanged. A CLI-only bundle copies
    its already configured headers/timeout/tool policy, not credentials or OAuth.
    """
    from openkb.stage_config import resolve_stage_reasoning

    effort = resolve_stage_reasoning(config, stage)
    if effort is None:
        return {}
    if bundle is None:
        from openkb.config import (
            LlmCredentialBundle,
            get_extra_headers,
            get_parallel_tool_calls,
            get_timeout,
        )

        parallel, explicit = get_parallel_tool_calls()
        bundle = LlmCredentialBundle(
            extra_headers=get_extra_headers(),
            timeout=get_timeout(),
            parallel_tool_calls=parallel,
            parallel_tool_calls_explicit=explicit,
        )
    return {"bundle": with_reasoning_effort(bundle, effort)}


def reasoning_fallback_allowed(error, model, effort, *, stream=False):
    """Only ChatGPT none -> low on a parameter-specific 400; never downgrade.

    Streaming is deliberately excluded, including errors before its first event:
    retrying a Runner could repeat already executed tools or earlier turns.
    Unset effort preserves historical provider behavior without a new retry.
    """
    if stream or effort != "none" or not model.startswith("chatgpt/"):
        return False
    if getattr(error, "status_code", None) != 400:
        return False
    text = str(error).lower()
    seen = set()
    item = error
    while item is not None and id(item) not in seen:
        seen.add(id(item))
        if getattr(item, "status_code", None) in (401, 403, 429):
            return False
        if any(
            word in str(item).lower()
            for word in ("quota", "auth", "content", "billing", "permission", "usage_limit_reached")
        ):
            return False
        item = item.__cause__ or item.__context__
    return any(key in text for key in ("reasoning_effort", "reasoning.effort")) and any(
        marker in text for marker in ("unsupported", "not supported")
    )


def _fallback_log():
    import logging

    logging.getLogger(__name__).warning(
        "reasoning requested=none effective=low fallback=true scope=current_request"
    )


def _success_log(effort):
    if effort is None:
        return
    import logging

    safe = effort if effort in ("none", "minimal", "low", "medium", "high", "xhigh") else "invalid"
    logging.getLogger(__name__).info(
        "reasoning requested=%s effective=%s fallback=false scope=current_request", safe, safe
    )


def reasoning_request_kwargs(model, effort, kwargs):
    """Keep explicit effort on unknown ChatGPT IDs despite legacy SDK metadata."""
    sent = dict(kwargs)
    if effort is not None:
        sent["reasoning_effort"] = effort
        if model.startswith(("chatgpt/", "openai/responses/")):
            allowed = list(sent.get("allowed_openai_params") or [])
            if "reasoning_effort" not in allowed:
                allowed.append("reasoning_effort")
            sent["allowed_openai_params"] = allowed
    return sent


def call_with_reasoning_fallback(call, *, model, messages, reasoning_effort=None, **kwargs):
    """Return the actual response; at most one additional nonstream leaf call."""
    sent = reasoning_request_kwargs(model, reasoning_effort, kwargs)
    try:
        response = call(model=model, messages=messages, **sent)
        _success_log(reasoning_effort)
        return response
    except Exception as error:
        if not reasoning_fallback_allowed(
            error, model, reasoning_effort, stream=sent.get("stream", False)
        ):
            raise
    _fallback_log()
    sent["reasoning_effort"] = "low"
    return call(model=model, messages=messages, **sent)


async def acall_with_reasoning_fallback(call, *, model, messages, reasoning_effort=None, **kwargs):
    """Async counterpart; never catches errors raised while consuming streams."""
    sent = reasoning_request_kwargs(model, reasoning_effort, kwargs)
    try:
        response = await call(model=model, messages=messages, **sent)
        _success_log(reasoning_effort)
        return response
    except Exception as error:
        if not reasoning_fallback_allowed(
            error, model, reasoning_effort, stream=sent.get("stream", False)
        ):
            raise
    _fallback_log()
    sent["reasoning_effort"] = "low"
    return await call(model=model, messages=messages, **sent)
