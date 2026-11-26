"""Indexing effort applies to descriptions as well as section summaries."""

import sys
from types import SimpleNamespace


def test_worker_description_inherits_explicit_indexing_effort(monkeypatch):
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
            choices=[SimpleNamespace(message=SimpleNamespace(content="description"))],
        )

    monkeypatch.setitem(sys.modules, "pageindex.utils", utils)
    monkeypatch.setattr(pageindex, "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=provider))
    with completion_policy({"summary_reasoning_effort": "low"}):
        assert utils._completion("chatgpt/gpt-6-luna", [], {}, False) == "description"
    assert len(seen) == 1
    assert seen[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" in seen[0]["allowed_openai_params"]
