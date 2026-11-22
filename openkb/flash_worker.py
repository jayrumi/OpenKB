"""Standalone Flash entry point. Run only with the separately pinned worker Python.

stdin/stdout are a single JSON request/result. SDK output is suppressed because
provider diagnostics can expose source text or authentication material.
"""

from __future__ import annotations

import contextlib

# `python -I /absolute/path/flash_worker.py` intentionally has no OpenKB on sys.path.
import importlib
import json
import logging
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
_policy = importlib.import_module("openkb.indexing_policy" if __package__ else "indexing_policy")
_structure = importlib.import_module("openkb.flash_structure" if __package__ else "flash_structure")
_reasoning = importlib.import_module("openkb.reasoning" if __package__ else "reasoning")
FlashAbort = _policy.FlashAbort
bounded_completion = _policy.bounded_completion
quota_error = _policy.quota_error


@contextlib.contextmanager
def completion_policy(request):
    """Pinned 0.7.4 private seam; replace both SDK retry loops in this worker only."""
    import asyncio

    import litellm
    import pageindex.utils as utils

    original = utils._acompletion
    original_sync = utils._completion
    calls = 0
    tokens = 0
    stopped = None
    diagnostics = {"calls": 0, "retries": 0}
    if request.get("summary_reasoning_effort") is not None:
        diagnostics.update(
            requested_reasoning_effort=request["summary_reasoning_effort"],
            effective_reasoning_effort=request["summary_reasoning_effort"],
            reasoning_fallbacks=0,
        )
    active = set()

    def reasoning_fallback(requested, effective):
        diagnostics.setdefault("requested_reasoning_effort", requested)
        diagnostics["effective_reasoning_effort"] = effective
        diagnostics["reasoning_fallbacks"] = diagnostics.get("reasoning_fallbacks", 0) + 1

    async def tracked(*, _retry=False, **kwargs):
        nonlocal calls, tokens
        if stopped is not None:
            raise stopped
        max_calls, max_tokens = request.get("max_calls"), request.get("max_tokens")
        if (max_calls is not None and calls >= max_calls) or (
            max_tokens is not None and tokens >= max_tokens
        ):
            raise FlashAbort("budget")
        calls += 1
        diagnostics["calls"] = calls
        diagnostics["retries"] += int(_retry)
        response = await litellm.acompletion(**kwargs)
        usage = getattr(response, "usage", None)
        tokens += getattr(usage, "total_tokens", 0) or 0
        if max_tokens is not None and tokens > max_tokens:
            raise FlashAbort("budget")
        return response

    async def complete(model, messages, sent):
        nonlocal stopped
        if stopped is not None:
            raise stopped
        task = asyncio.current_task()
        active.add(task)
        attempts = 0
        sent = _reasoning.reasoning_request_kwargs(
            model, request.get("summary_reasoning_effort"), sent
        )

        async def leaf(**kwargs):
            nonlocal attempts
            retry = attempts > 0
            attempts += 1
            return await tracked(_retry=retry, **kwargs)

        try:
            response = await bounded_completion(
                leaf,
                model,
                messages,
                utils._claude_none_kwargs(model, {"drop_params": True, **sent}),
                timeout=request.get("request_timeout", 120),
                retries=request.get("retries", 2),
                backoff=request.get("backoff", 2),
                on_reasoning_fallback=reasoning_fallback,
            )
            utils._add_usage(response)
            return response.choices[0].message.content
        except FlashAbort as error:
            stopped = error
            for sibling in active:
                if sibling is not task and not sibling.done():
                    sibling.cancel()
            raise
        except (AttributeError, IndexError):
            stopped = FlashAbort("request")
            raise stopped from None
        finally:
            active.discard(task)

    def complete_sync(model, messages, sent, return_finish_reason=False):
        content = asyncio.run(complete(model, messages, sent))
        return (content, "finished") if return_finish_reason else content

    utils._acompletion = complete
    utils._completion = complete_sync
    try:
        yield diagnostics
        # SDK gather(return_exceptions=True) must never turn a fatal leaf into success.
        if stopped is not None:
            raise stopped
    finally:
        utils._acompletion = original
        utils._completion = original_sync


def execute(request, *, client_factory=None):
    """Use the SDK's public API; latest storage never touches legacy storage."""
    if version("pageindex") != "0.7.4":
        raise ValueError("sdk")
    if any(
        model.startswith("claude-code/")
        for model in (request["model"], request.get("summary_model", request["model"]))
    ):
        raise ValueError("This backend bypasses bounded LiteLLM policy")
    injected_client = client_factory is not None
    if client_factory is None:
        from pageindex import PageIndexClient

        client_factory = PageIndexClient
    with tempfile.TemporaryDirectory(prefix="openkb-flash-") as storage:
        client = client_factory(
            mode="local",
            index_model=request["model"],
            summary_model=request.get("summary_model", request["model"]),
            storage_path=storage,
            summary_concurrency=request.get("summary_concurrency", 1),
            optimize="off",
            **(
                {"summary_reasoning_effort": request["summary_reasoning_effort"]}
                if request.get("summary_reasoning_effort") is not None
                else {}
            ),
        )
        # Fake clients do not implement the vendor layout pipeline. Real workers
        # repair its layout before the SDK generates any model summaries.
        structure_policy = (
            contextlib.nullcontext() if injected_client else _structure.flash_structure_policy()
        )
        with structure_policy:
            submitted = client.submit_document(request["pdf_path"], mode="flash")
        doc_id = submitted["doc_id"]
        metadata = client.get_document(doc_id)
        if metadata.get("status") != "completed":
            raise ValueError("Flash document is not completed")
        tree = client.get_tree(doc_id, node_summary=True, include_text=True)
        if tree.get("status") != "completed":
            raise ValueError("Flash tree is not completed")
        return {
            "sdk_version": "0.7.4",
            "doc_name": metadata["name"],
            "doc_description": metadata.get("description") or "",
            "page_count": metadata["pageNum"],
            "structure": tree["result"],
            "usage": submitted.get("usage", {}),
        }


def main():
    error = None
    result = None
    logging.disable(logging.CRITICAL)
    try:
        request = json.loads(sys.stdin.read())
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            with completion_policy(request) as diagnostics:
                result = execute(request)
                result["diagnostics"] = dict(diagnostics)
    except BaseException as exc:
        if quota_error(exc) or "quota" in str(exc).lower():
            error = "quota"
        elif isinstance(exc, FlashAbort) and str(exc) == "budget":
            error = "budget"
        elif getattr(exc, "status_code", None) in (401, 403) or any(
            word in str(exc).lower() for word in ("oauth", "authentication", "api key", "auth")
        ):
            error = "auth"
        elif str(exc) == "sdk":
            error = "sdk"
        else:
            error = "request"
    sys.stdout.write(json.dumps({"error": error} if error else result, ensure_ascii=False))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
