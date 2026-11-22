"""Real layout and real SDK orchestration with networking forbidden, not inference."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("OPENKB_FLASH_TEST_PYTHON"), reason="isolated Flash Python not selected"
)
def test_actual_sdk_layout_and_worker_with_mocked_llm(tmp_path):
    import fitz

    from openkb.flash_adapter import validate_result

    pdf = tmp_path / "layout.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "1 Introduction", fontsize=20)
        page.insert_textbox(
            fitz.Rect(72, 110, 520, 750),
            ("This chapter discusses deterministic indexing and safe retrieval. " * 40),
            fontsize=10,
        )
        document.save(pdf)
    source = Path(__file__).resolve().parents[1] / "openkb"
    script = """
import socket, sys, json, asyncio, os
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
from types import SimpleNamespace
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('NETWORK FORBIDDEN'))
sys.path.insert(0, sys.argv[1])
import flash_worker
import litellm
import pageindex.utils as utils
from pageindex.flash import page_index_flash
# Disable provider metadata downloads; all model calls below are explicit local fakes.
utils.start_learning_if_effort_dropped = lambda *a, **kw: True
utils.learn_model_if_effort_dropped = lambda *a, **kw: None
calls = []
requests = []
async def fake(**kw):
    calls.append(kw['model'])
    requests.append(kw['messages'])
    assert kw['messages'][0]['role'] == 'system'
    assert 'SUMMARY ATTRIBUTION POLICY' in kw['messages'][0]['content']
    usage = SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3)
    message = SimpleNamespace(content='{"summary": "Mock section summary"}')
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])
litellm.acompletion = fake
layout = page_index_flash(sys.argv[2], summary=False, optimize=False)
assert layout['structure']
with flash_worker.completion_policy({'max_calls': 10}):
    result = flash_worker.execute({'pdf_path': sys.argv[2], 'model': 'chatgpt/gpt-6-luna'})
assert len(calls) >= 2 and all(model == 'chatgpt/gpt-6-luna' for model in calls)
assert any('one-sentence description' in str(messages[1:]) for messages in requests)
assert any('one-sentence description' not in str(messages[1:]) for messages in requests)
print('RESULT_JSON=' + json.dumps(result))
"""
    completed = subprocess.run(
        [os.environ["OPENKB_FLASH_TEST_PYTHON"], "-I", "-c", script, str(source), str(pdf)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.split("RESULT_JSON=")[-1])
    assert validate_result(payload)["page_count"] == 1
