"""Serialized SDK requests, fake auth before routing, no model/network calls."""

import json

import pytest
import test_codex_stream_transport as transport

RequestCaptured = transport.RequestCaptured
cached_wire = transport.cached_wire
wire = transport.wire


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model,effort", [("gpt-6-sol", "medium"), ("gpt-6-luna", "low")])
async def test_compiler_stage_effort_serialized(cached_wire, model, effort, asynchronous):
    from openkb.agent import compiler
    from openkb.config import LlmCredentialBundle

    args = (f"chatgpt/{model}", [{"role": "user", "content": "Offline fixture"}], "wire")
    bundle = LlmCredentialBundle(reasoning_effort=effort)
    result = (
        await compiler._llm_call_async(*args, bundle=bundle)
        if asynchronous
        else compiler._llm_call(*args, bundle=bundle)
    )
    assert result == "Offline lifecycle answer."
    assert len(cached_wire.requests) == 1
    body = json.loads(cached_wire.requests[0].content)
    assert body["model"] == model
    assert body["reasoning"]["effort"] == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model,effort", [("gpt-6-sol", "medium"), ("gpt-6-luna", "low")])
async def test_query_medium_reaches_serialized_responses(wire, kb_dir, stream, model, effort):
    from agents import RunConfig, Runner

    from openkb.agent.query import build_query_agent, iter_agent_response_events
    from openkb.config import LlmCredentialBundle

    agent = build_query_agent(
        str(kb_dir / "wiki"),
        f"chatgpt/{model}",
        bundle=LlmCredentialBundle(reasoning_effort=effort),
    )
    try:
        if stream:
            async for _ in iter_agent_response_events(
                agent, "offline fixture", run_config=RunConfig(tracing_disabled=True)
            ):
                pass
        else:
            await Runner.run(agent, "offline fixture", run_config=RunConfig(tracing_disabled=True))
    except RequestCaptured:
        pass
    assert len(wire.requests) == 1
    body = json.loads(wire.requests[0].content)
    assert body["model"] == model
    assert body["reasoning"]["effort"] == effort


@pytest.mark.asyncio
async def test_leaf_low_reaches_sdk_http(wire):
    import litellm

    from openkb.provider_compat import install_response_compat
    from openkb.reasoning import acall_with_reasoning_fallback

    install_response_compat()
    try:
        await acall_with_reasoning_fallback(
            litellm.acompletion,
            model="chatgpt/gpt-6-luna",
            messages=[{"role": "user", "content": "offline"}],
            reasoning_effort="low",
            max_retries=0,
        )
    except RequestCaptured:
        pass
    assert len(wire.requests) == 1
    assert json.loads(wire.requests[0].content)["reasoning"]["effort"] == "low"


@pytest.mark.parametrize("effort", ["low", "none"])
def test_flash_sdk_explicit_low_serialized_http(tmp_path, effort):
    import os
    import subprocess
    from pathlib import Path

    import fitz

    python = os.environ.get("OPENKB_FLASH_TEST_PYTHON")
    if not python:
        pytest.skip("isolated Flash Python required")
    pdf = tmp_path / "offline.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "1 Offline section", fontsize=20)
        page.insert_textbox(
            fitz.Rect(72, 110, 520, 750), "Offline deterministic fixture. " * 60, fontsize=10
        )
        doc.save(pdf)
    script = r"""
import os, sys, socket, json, asyncio
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('NETWORK FORBIDDEN'))
sys.path.insert(0, sys.argv[1])
import litellm, httpx
from litellm.llms.chatgpt.authenticator import Authenticator
Authenticator.__init__ = lambda self: None
Authenticator.get_access_token = lambda self: 'offline-token'
Authenticator.get_account_id = lambda self: None
Authenticator.get_api_base = lambda self: 'https://codex.invalid'
def forbidden(*a, **k):
    raise AssertionError('AUTH FILE ACCESS FORBIDDEN')
Authenticator._read_auth_file = forbidden
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
ChatGPTResponsesAPIConfig.validate_environment = lambda *a, **k: {}
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx import llm_http_handler
requests = []
class Captured(BaseException):
    pass
async def capture(request):
    requests.append(json.loads(request.content))
    if sys.argv[3] == 'none' and len(requests) == 1:
        return httpx.Response(400, json={'error': {
            'message': "Unsupported value: 'none' for reasoning.effort",
            'param': 'reasoning.effort', 'code': 'unsupported_value'}})
    raise Captured()
original_create = AsyncHTTPHandler.create_client
def create(self, *a, **k):
    return httpx.AsyncClient(transport=httpx.MockTransport(capture))
AsyncHTTPHandler.create_client = create
import flash_worker
import pageindex.utils as utils
utils.start_learning_if_effort_dropped = lambda *a, **kw: True
utils.learn_model_if_effort_dropped = lambda *a, **kw: None
try:
    with flash_worker.completion_policy({'retries': 0, 'max_calls': 3}):
        flash_worker.execute({'pdf_path': sys.argv[2], 'model': 'chatgpt/gpt-6-luna',
                              'summary_reasoning_effort': sys.argv[3]})
except BaseException as e:
    print('STOP_TYPE=' + type(e).__name__)
assert requests, 'SDK never reached HTTP'
expected = ['none', 'low'] if sys.argv[3] == 'none' else ['low']
assert [r.get('reasoning', {}).get('effort') for r in requests] == expected, requests
assert requests[0]['model'] == 'gpt-6-luna'
print('WIRE_OK')
"""
    source = str(Path(__file__).resolve().parents[1] / "openkb")
    run = subprocess.run(
        [python, "-I", "-c", script, source, str(pdf), effort],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "WIRE_OK" in run.stdout
