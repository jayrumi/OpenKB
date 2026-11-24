def test_api_run_config_preserves_effort_allowance():
    from openkb.agent.query import build_run_config_from_bundle
    from openkb.config import LlmCredentialBundle

    config = build_run_config_from_bundle(
        "openai/responses/gpt-6-sol", LlmCredentialBundle(reasoning_effort="medium")
    )
    assert config.model_settings.extra_args == {"allowed_openai_params": ["reasoning_effort"]}
