"""Offline regression for exact Codex IDs absent from legacy LiteLLM metadata."""
import importlib

from openkb.provider_compat import install_response_compat


def test_exact_codex_model_routes_through_responses_without_rewriting_id(monkeypatch):
    module = importlib.import_module("litellm.main")
    monkeypatch.setattr(module, "_get_model_info_helper", lambda **kwargs: {})
    install_response_compat()
    info, model = module.responses_api_bridge_check("gpt-6.2-sol", "chatgpt")
    assert info["mode"] == "responses"
    assert model == "gpt-6.2-sol"


def test_codex_transport_patch_does_not_change_other_provider_routing(monkeypatch):
    module = importlib.import_module("litellm.main")
    monkeypatch.setattr(module, "_get_model_info_helper", lambda **kwargs: {})
    install_response_compat()
    info, model = module.responses_api_bridge_check("some-other-model", "anthropic")
    assert info.get("mode") != "responses"
    assert model == "some-other-model"
