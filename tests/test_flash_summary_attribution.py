"""Offline request-contract tests; mocked outputs do not prove model quality."""

import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("version", [-1, True, None])
def test_worker_rejects_unknown_summary_policy_before_discovery(version, monkeypatch):
    import pageindex

    from openkb.flash_worker import completion_policy

    utils = SimpleNamespace(_acompletion=object(), _completion=object())
    monkeypatch.setitem(sys.modules, "pageindex.utils", utils)
    monkeypatch.setattr(pageindex, "utils", utils, raising=False)
    with pytest.raises(ValueError, match="summary policy"):
        with completion_policy({"summary_policy_version": version}):
            pass


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        "We shipped a code review surface (called Antique), code search, and an IAM system.",
        (
            "Scion supports directory permissions. Furthermore, "
            "code review tooling has approval constraints."
        ),
    ],
)
def test_worker_requests_preserve_capability_subjects_for_both_summary_paths(
    monkeypatch, sync, source
):
    import pageindex

    from openkb.flash_worker import completion_policy

    utils = SimpleNamespace(
        _acompletion=object(),
        _completion=object(),
        _claude_none_kwargs=lambda model, kwargs: kwargs,
        _add_usage=lambda response: None,
    )
    seen = []

    async def provider(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"mock output"}'))],
        )

    monkeypatch.setitem(sys.modules, "pageindex.utils", utils)
    monkeypatch.setattr(pageindex, "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=provider))
    messages = [{"role": "user", "content": "Return the requested summary JSON. Source: " + source}]
    before = deepcopy(messages)
    original_async, original_sync = utils._acompletion, utils._completion
    with completion_policy({"summary_reasoning_effort": "low", "max_calls": 1}):
        if sync:
            output = utils._completion("chatgpt/gpt-6-luna", messages, {}, False)
        else:
            output = asyncio.run(utils._acompletion("chatgpt/gpt-6-luna", messages, {}))
    assert output == '{"summary":"mock output"}'
    assert len(seen) == 1
    sent = seen[0]["messages"]
    assert sent[0]["role"] == "system"
    policy = sent[0]["content"]
    assert "SUMMARY ATTRIBUTION POLICY" in policy
    assert "capability" in policy and "subject" in policy
    assert "permissions" in policy and "approval" in policy
    assert "code review" in policy and "code search" in policy and "identity" in policy
    assert "explicitly combined" in policy
    assert "output format" in policy and "unclear" in policy
    assert "Antique" not in policy and "Scion" not in policy
    assert sent[1:] == before and messages == before
    assert seen[0]["reasoning_effort"] == "low"
    assert utils._acompletion is original_async and utils._completion is original_sync
