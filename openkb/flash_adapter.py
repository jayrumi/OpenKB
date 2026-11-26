"""Bridge an opt-in, isolated Flash SDK to OpenKB's pinned legacy reader."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from openkb.locks import atomic_write_text
from openkb.summary_policy import SUMMARY_POLICY_VERSION

logger = logging.getLogger(__name__)
SDK_VERSION = "0.7.4"
CACHE_SCHEMA = 4


def pdf_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(result: dict) -> dict:
    """Reject incomplete, incompatible, or malformed worker/cache payloads."""
    if not isinstance(result, dict) or result.get("sdk_version") != SDK_VERSION:
        raise ValueError("Flash SDK version mismatch")
    count = result.get("page_count")
    if type(count) is not int or count < 1:
        raise ValueError("Invalid Flash page count")
    for field in ("doc_name", "doc_description"):
        if not isinstance(result.get(field), str):
            raise ValueError(f"Invalid Flash {field}")
    structure = result.get("structure")
    if not isinstance(structure, list) or not structure:
        raise ValueError("Empty Flash structure")
    stack = [(node, 1, count) for node in structure]
    ids = set()
    while stack:
        node, lower, upper = stack.pop()
        if not isinstance(node, dict):
            raise ValueError("Invalid Flash structure node")
        start, end = node.get("start_index"), node.get("end_index")
        if type(start) is not int or type(end) is not int or not lower <= start <= end <= upper:
            raise ValueError("Invalid Flash structure page range")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            raise ValueError("Invalid Flash structure node ID")
        ids.add(node_id)
        if not isinstance(node.get("title"), str) or not isinstance(node.get("summary"), str):
            raise ValueError("Incomplete Flash structure summary")
        if "text" in node and not isinstance(node["text"], str):
            raise ValueError("Invalid Flash node text")
        children = node.get("nodes", [])
        if not isinstance(children, list):
            raise ValueError("Invalid Flash child structure")
        stack.extend((child, start, end) for child in children)
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Invalid Flash usage")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid Flash token usage")
    diagnostics = result.get("diagnostics", {"calls": 0, "retries": 0})
    if (
        not isinstance(diagnostics, dict)
        or any(
            type(diagnostics.get(key)) is not int or diagnostics[key] < 0
            for key in ("calls", "retries")
        )
        or diagnostics["retries"] > diagnostics["calls"]
    ):
        raise ValueError("Invalid Flash diagnostics")
    clean_diagnostics = {key: diagnostics[key] for key in ("calls", "retries")}
    reasoning_keys = (
        "requested_reasoning_effort",
        "effective_reasoning_effort",
        "reasoning_fallbacks",
    )
    if any(key in diagnostics for key in reasoning_keys):
        from openkb.stage_config import REASONING_EFFORTS

        if any(diagnostics.get(key) not in REASONING_EFFORTS for key in reasoning_keys[:2]):
            raise ValueError("Invalid Flash reasoning diagnostics")
        count_fallbacks = diagnostics.get("reasoning_fallbacks")
        if type(count_fallbacks) is not int or not 0 <= count_fallbacks <= diagnostics["retries"]:
            raise ValueError("Invalid Flash reasoning fallback count")
        clean_diagnostics.update({key: diagnostics[key] for key in reasoning_keys})
    # Do not persist or log arbitrary SDK fields (which may contain prompts).
    return (
        {
            key: result[key]
            for key in ("sdk_version", "doc_name", "doc_description", "structure", "page_count")
        }
        | {"diagnostics": clean_diagnostics}
        | {
            "usage": {
                k: usage[k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if k in usage
            }
        }
    )


def run_worker(python: str, request: dict, timeout: float) -> dict:
    """A single process, never a document-level retry; kill its whole process group."""
    if not Path(python).is_absolute() or not os.access(python, os.X_OK):
        raise ValueError("flash_python must be an absolute executable path")
    worker = Path(__file__).with_name("flash_worker.py")
    process = subprocess.Popen(
        [python, "-I", str(worker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(json.dumps(request), timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    try:
        response = json.loads(stdout)
    except (ValueError, TypeError):
        raise RuntimeError("Flash worker failed without a valid result") from None
    if not isinstance(response, dict):
        raise RuntimeError("Flash worker returned an invalid response")
    if process.returncode or response.get("error"):
        # Whitelisted messages only: never surface raw provider exceptions.
        reason = response.get("error")
        allowed = {"quota", "auth", "request", "sdk", "budget", "configuration"}
        raise RuntimeError("Flash indexing blocked: " + (reason if reason in allowed else "worker"))
    return response


def index_pdf(pdf_path: Path, config: dict) -> dict:
    """Return a validated successful index, recovered outside mutation snapshots."""
    python = config.get("flash_python")
    if not isinstance(python, str) or not python:
        raise ValueError("indexing_backend: flash requires flash_python")
    from openkb.stage_config import resolve_stage_model, resolve_stage_reasoning

    model = resolve_stage_model(config, "indexing")
    if not isinstance(model, str) or not model:
        raise ValueError("Flash indexing model must be a nonempty string")
    timeout = config.get("flash_worker_timeout", 3600)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("flash_worker_timeout must be positive and finite")
    concurrency = config.get("indexing_concurrency", 1)
    retries = config.get("indexing_retries", 2)
    request_timeout = config.get("indexing_timeout", 120)
    backoff = config.get("indexing_backoff", 2)
    if type(concurrency) is not int or not 1 <= concurrency <= 2:
        raise ValueError("indexing_concurrency must be an integer from 1 to 2")
    if type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("indexing_retries must be an integer from 0 to 5")
    for key, value, minimum, maximum in (
        ("indexing_timeout", request_timeout, 0, 600),
        ("indexing_backoff", backoff, 0, 60),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < minimum
            or value > maximum
            or (key == "indexing_timeout" and value == 0)
        ):
            raise ValueError(f"{key} is outside the supported finite range")
    policy = {
        "summary_policy_version": SUMMARY_POLICY_VERSION,
        "model": model,
        "summary_model": model,
        "summary_concurrency": concurrency,
        "optimize": "off",
        "request_timeout": request_timeout,
        "retries": retries,
        "backoff": backoff,
        "max_calls": config.get("flash_max_calls"),
        "max_tokens": config.get("flash_max_tokens"),
    }
    effort = resolve_stage_reasoning(config, "indexing")
    if effort is not None:
        policy["summary_reasoning_effort"] = effort
    for name in ("max_calls", "max_tokens"):
        if policy[name] is not None and (type(policy[name]) is not int or policy[name] < 1):
            raise ValueError(f"Flash {name} must be a positive integer")
    identity = {
        "schema": CACHE_SCHEMA,
        "sha256": pdf_hash(pdf_path),
        "sdk_version": SDK_VERSION,
        "policy": policy,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(
        config.get("flash_cache_dir")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "openkb" / "flash"
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache = root / f"{key}.json"
    start = time.monotonic()
    import portalocker

    with portalocker.Lock(str(cache.with_suffix(".lock")), mode="a", timeout=timeout):
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload["identity"] != identity:
                raise ValueError("Cache identity mismatch")
            result = validate_result(payload["result"])
            cached = True
        except (OSError, ValueError, KeyError, TypeError):
            result = validate_result(
                run_worker(
                    python,
                    {
                        **policy,
                        "pdf_path": str(pdf_path.resolve()),
                        "sdk_version": SDK_VERSION,
                    },
                    timeout,
                )
            )
            atomic_write_text(
                cache, json.dumps({"identity": identity, "result": result}, ensure_ascii=False)
            )
            cached = False
    logger.info(
        "stage=indexing backend=flash model=%s duration_s=%.3f "
        "cache_hit=%s calls=%d retries=%d usage=%s",
        model,
        time.monotonic() - start,
        cached,
        result["diagnostics"]["calls"],
        result["diagnostics"]["retries"],
        result["usage"],
    )
    return result
