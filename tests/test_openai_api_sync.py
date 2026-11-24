"""Nonstream compiler API serialization behind blocked HTTP transport."""

import json

import httpx
import pytest


@pytest.mark.parametrize("model,effort", [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")])
def test_sync_compiler_api_wire(monkeypatch, model, effort):
    import litellm
    from litellm.main import completion

    from openkb.agent.compiler import _llm_call
    from openkb.config import LlmCredentialBundle

    class Captured(BaseException):
        pass

    requests = []

    def capture(self, request, *a, **kw):
        requests.append(request)
        raise Captured()

    monkeypatch.setattr(httpx.Client, "send", capture)
    monkeypatch.setattr(litellm, "completion", completion)
    with pytest.raises(Captured):
        _llm_call(
            f"openai/responses/{model}",
            [{"role": "user", "content": "offline"}],
            "wire",
            bundle=LlmCredentialBundle(api_key="offline", reasoning_effort=effort),
        )
    assert len(requests) == 1
    assert requests[0].url.host == "api.openai.com"
    body = json.loads(requests[0].content)
    assert body["model"] == model and body["reasoning"]["effort"] == effort
