"""Stage overrides round-trip through existing config REST contracts."""

import pytest
import yaml

from openkb.api_config import apply_kb_config_patch, read_global_config, read_kb_config
from openkb.api_models import KbConfigPatchRequest


@pytest.fixture(autouse=True)
def isolate_global_config(tmp_path, monkeypatch):
    monkeypatch.setattr("openkb.config.GLOBAL_CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr("openkb.config.GLOBAL_CONFIG_PATH", tmp_path / "global" / "global.yaml")


@pytest.mark.parametrize("stage", ["indexing", "compilation", "query"])
def test_stage_models_patch_read_and_clear(kb_dir, stage):
    key = f"{stage}_model"
    request = KbConfigPatchRequest(kb="test", config={key: "chatgpt/exact-token"})
    apply_kb_config_patch(kb_dir, request)
    assert getattr(read_kb_config(kb_dir), key) == "chatgpt/exact-token"
    apply_kb_config_patch(kb_dir, KbConfigPatchRequest(kb="test", config={key: None}))
    assert getattr(read_kb_config(kb_dir), key) is None
    assert key not in yaml.safe_load((kb_dir / ".openkb" / "config.yaml").read_text())


def test_global_stage_models_are_reported_with_raw_global_values(kb_dir, tmp_path):
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "global.yaml").write_text("query_model: chatgpt/gpt-6.2-sol\n")
    assert read_global_config().query_model == "chatgpt/gpt-6.2-sol"
    response = read_kb_config(kb_dir)
    assert response.query_model == "chatgpt/gpt-6.2-sol"
    assert response.global_values.query_model == "chatgpt/gpt-6.2-sol"
    assert response.sources["query_model"] == "global"
