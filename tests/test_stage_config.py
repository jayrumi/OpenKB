"""Stage model selection stays request-local and preserves provider tokens."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from openkb.config import DEFAULT_CONFIG


@pytest.mark.parametrize("stage", ["indexing", "compilation", "query"])
def test_stage_model_override_and_legacy_fallback(stage):
    from openkb.stage_config import resolve_stage_model

    config = {"model": "legacy/exact-token", f"{stage}_model": "chatgpt/gpt-6-luna"}
    before = dict(config)
    assert resolve_stage_model(config, stage) == "chatgpt/gpt-6-luna"
    assert config == before
    assert resolve_stage_model({"model": "legacy/exact-token"}, stage) == "legacy/exact-token"
    assert resolve_stage_model({}, stage) == DEFAULT_CONFIG["model"]


@pytest.mark.parametrize("value", [None])
def test_unset_stage_model_uses_legacy_model(value):
    from openkb.stage_config import resolve_stage_model

    assert resolve_stage_model({"model": "legacy", "query_model": value}, "query") == "legacy"


@pytest.mark.parametrize("value", [False, 42, [], {}, "", "   "])
def test_invalid_stage_model_is_rejected_without_normalizing(value):
    from openkb.stage_config import resolve_stage_model

    with pytest.raises(ValueError, match="query_model"):
        resolve_stage_model({"query_model": value}, "query")


def test_stage_model_tokens_are_verbatim_and_selection_is_isolated():
    from openkb.stage_config import resolve_stage_model

    configs = [{"query_model": f"chatgpt/exact-token-{i}"} for i in range(20)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda config: resolve_stage_model(config, "query"), configs))
    assert results == [config["query_model"] for config in configs]
    assert resolve_stage_model({"query_model": " provider/token "}, "query") == " provider/token "


def test_unknown_stage_is_rejected():
    from openkb.stage_config import resolve_stage_model

    with pytest.raises(ValueError, match="stage"):
        resolve_stage_model({}, "typo")
