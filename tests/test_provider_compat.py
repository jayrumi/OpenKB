"""Offline reproductions of the two production-local provider compatibility fixes."""

from unittest.mock import patch

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage


def _items():
    return [
        ResponseOutputMessage(
            id="msg_test",
            type="message",
            role="assistant",
            status="completed",
            content=[{"type": "output_text", "text": "Checking wiki.", "annotations": []}],
        ),
        ResponseFunctionToolCall(
            id="fc_test",
            type="function_call",
            call_id="call_test",
            name="read_file",
            arguments='{"path":"index.md"}',
            status="completed",
        ),
    ]


def test_sol_commentary_and_tools_remain_one_choice_without_mutating_original():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler as Handler,
    )

    from openkb.provider_compat import coalesce_response_choices

    choices = Handler._convert_response_output_to_choices(_items())
    original_length = len(choices)
    fixed = coalesce_response_choices(choices)
    assert len(fixed) == 1
    assert fixed[0].message.content == "Checking wiki."
    assert fixed[0].message.tool_calls[0].function.name == "read_file"
    assert fixed[0].finish_reason == "tool_calls"
    assert len(choices) == original_length


def test_sol_compat_install_is_idempotent_and_preserves_text_only(monkeypatch):
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler as Handler,
    )

    from openkb.provider_compat import install_response_compat

    original = Handler._convert_response_output_to_choices
    monkeypatch.setattr(Handler, "_convert_response_output_to_choices", staticmethod(original))
    install_response_compat()
    installed = Handler._convert_response_output_to_choices
    install_response_compat()
    assert Handler._convert_response_output_to_choices is installed
    choices = installed(_items())
    assert len(choices) == 1
    assert choices[0].message.tool_calls[0].function.name == "read_file"
    text = installed(_items()[:1])
    assert text[0].message.content == "Checking wiki."
    assert text[0].finish_reason == "stop"


def test_pageindex_chatgpt_validation_does_not_require_fake_api_key(monkeypatch):
    from openkb.provider_compat import OAuthPageIndexClient

    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    with patch(
        "pageindex.PageIndexClient._validate_llm_provider",
        side_effect=AssertionError("legacy validation"),
    ):
        OAuthPageIndexClient._validate_llm_provider("chatgpt/gpt-6-luna")
    import os

    assert "CHATGPT_API_KEY" not in os.environ


def test_pageindex_other_providers_retain_validation():
    from openkb.provider_compat import OAuthPageIndexClient

    with patch(
        "pageindex.PageIndexClient._validate_llm_provider",
        side_effect=ValueError("missing real key"),
    ) as validate:
        with pytest.raises(ValueError, match="missing real key"):
            OAuthPageIndexClient._validate_llm_provider("openai/model")
    validate.assert_called_once_with("openai/model")


@pytest.mark.parametrize("builder", ["build_query_agent", "build_chat_agent"])
def test_query_and_chat_install_bridge_before_using_chatgpt(kb_dir, builder):
    from openkb.agent import query

    with patch("openkb.provider_compat.install_response_compat") as install:
        getattr(query, builder)(kb_dir, "chatgpt/gpt-6.2-sol")
    install.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compiler_installs_bridge_before_chatgpt_call(asynchronous):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openkb.agent import compiler

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    with (
        patch("openkb.provider_compat.install_response_compat") as install,
        patch.object(compiler.litellm, "completion", return_value=response),
        patch.object(
            compiler.litellm, "acompletion", new_callable=AsyncMock, return_value=response
        ),
        patch.object(compiler, "_Spinner"),
    ):
        args = ("chatgpt/gpt-6.2-sol", [{"role": "user", "content": "question"}], "test")
        result = (
            asyncio.run(compiler._llm_call_async(*args))
            if asynchronous
            else compiler._llm_call(*args)
        )
    assert result == "answer"
    install.assert_called_once()
