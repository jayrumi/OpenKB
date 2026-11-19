import json
import os
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest


def result():
    return {
        "sdk_version": "0.7.4",
        "doc_name": "book.pdf",
        "doc_description": "summary",
        "page_count": 1,
        "structure": [
            {
                "title": "Chapter",
                "node_id": "0000",
                "start_index": 1,
                "end_index": 1,
                "summary": "section",
                "text": "body",
            }
        ],
        "usage": {"total_tokens": 10},
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("indexing_concurrency", 2),
        ("indexing_timeout", 7),
        ("indexing_retries", 0),
        ("indexing_backoff", 5),
    ],
)
def test_configured_policy_is_forwarded_and_changes_cache(tmp_path, monkeypatch, key, value):
    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    runner = Mock(return_value=result())
    monkeypatch.setattr(adapter, "run_worker", runner)
    config = {
        "flash_python": sys.executable,
        "model": "exact",
        "flash_cache_dir": str(tmp_path / "cache"),
    }
    adapter.index_pdf(pdf, config)
    adapter.index_pdf(pdf, {**config, key: value})
    assert runner.call_count == 2
    names = {
        "indexing_concurrency": "summary_concurrency",
        "indexing_timeout": "request_timeout",
        "indexing_retries": "retries",
        "indexing_backoff": "backoff",
    }
    assert runner.call_args.args[1][names[key]] == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("indexing_concurrency", 3),
        ("indexing_concurrency", True),
        ("indexing_timeout", 0),
        ("indexing_timeout", float("nan")),
        ("indexing_retries", -1),
        ("indexing_retries", True),
        ("indexing_backoff", -1),
        ("indexing_backoff", float("inf")),
    ],
)
def test_invalid_policy_rejected_before_worker(tmp_path, monkeypatch, key, value):
    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    runner = Mock(return_value=result())
    monkeypatch.setattr(adapter, "run_worker", runner)
    with pytest.raises(ValueError, match=key):
        adapter.index_pdf(pdf, {"flash_python": sys.executable, key: value})
    runner.assert_not_called()


def test_diagnostics_are_validated_and_logged(tmp_path, monkeypatch, caplog):
    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(
        adapter,
        "run_worker",
        Mock(return_value={**result(), "diagnostics": {"calls": 3, "retries": 1}}),
    )
    config = {
        "flash_python": sys.executable,
        "model": "exact",
        "flash_cache_dir": str(tmp_path / "cache"),
    }
    import logging

    with caplog.at_level(logging.INFO):
        data = adapter.index_pdf(pdf, config)
    assert data["diagnostics"] == {"calls": 3, "retries": 1}
    assert "calls=3 retries=1" in caplog.text
    with pytest.raises(ValueError, match="diagnostics"):
        adapter.validate_result({**result(), "diagnostics": {"calls": -1, "retries": 0}})


def test_same_pdf_parallel_requests_spend_only_once(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    calls = []

    def runner(*args):
        calls.append(args)
        time.sleep(0.1)
        return result()

    monkeypatch.setattr(adapter, "run_worker", runner)
    config = {
        "flash_python": sys.executable,
        "model": "exact",
        "flash_cache_dir": str(tmp_path / "cache"),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: adapter.index_pdf(pdf, config), range(2)))
    assert results[0] == results[1]
    assert len(calls) == 1


def test_wall_timeout_kills_worker_process_group(tmp_path, monkeypatch):
    from openkb import flash_adapter as adapter

    monkeypatch.setattr(adapter, "__file__", str(tmp_path / "flash_adapter.py"))
    (tmp_path / "flash_worker.py").write_text("import time; time.sleep(100)")
    killed = []
    original_kill = os.killpg

    def kill(group, sig):
        killed.append((group, sig))
        original_kill(group, sig)

    monkeypatch.setattr(adapter.os, "killpg", kill)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.run_worker(sys.executable, {}, 0.1)
    assert len(killed) == 1 and killed[0][1] == signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(killed[0][0], 0)


def test_cancellation_kills_whole_worker_group(monkeypatch):
    from openkb import flash_adapter as adapter

    process = Mock(pid=12345)
    process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
    monkeypatch.setattr(adapter.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(adapter.os, "killpg", kill)
    with pytest.raises(KeyboardInterrupt):
        adapter.run_worker(sys.executable, {}, 1)
    kill.assert_called_once_with(12345, signal.SIGKILL)


def test_cached_complete_index_survives_retry_and_corruption(tmp_path, monkeypatch):
    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf-content")
    runner = Mock(return_value=result())
    monkeypatch.setattr(adapter, "run_worker", runner)
    config = {
        "flash_python": "/usr/bin/python3",
        "indexing_model": "chatgpt/gpt-6-luna",
        "flash_cache_dir": str(tmp_path / "outside-kb-cache"),
    }
    first = adapter.index_pdf(pdf, config)
    assert adapter.index_pdf(pdf, config) == first
    assert runner.call_count == 1
    assert runner.call_args.args[1]["model"] == "chatgpt/gpt-6-luna"
    cache = next((tmp_path / "outside-kb-cache").glob("*.json"))
    payload = json.loads(cache.read_text())
    payload["result"]["structure"] = []
    cache.write_text(json.dumps(payload))
    adapter.index_pdf(pdf, config)
    assert runner.call_count == 2
    adapter.index_pdf(pdf, {**config, "indexing_model": "other/exact"})
    assert runner.call_count == 3


def test_invalid_worker_result_is_not_cached(tmp_path, monkeypatch):
    from openkb import flash_adapter as adapter

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(adapter, "run_worker", lambda *a: {**result(), "structure": []})
    with pytest.raises(ValueError, match="structure"):
        adapter.index_pdf(
            pdf,
            {
                "flash_python": "/usr/bin/python3",
                "model": "exact",
                "flash_cache_dir": str(tmp_path / "cache"),
            },
        )
    assert not list((tmp_path / "cache").glob("*.json"))
