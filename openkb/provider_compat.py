"""Small, in-process compatibility shims; never rewrite installed dependencies.

PageIndex 0.3's key check predates LiteLLM's ChatGPT OAuth provider. LiteLLM
1.87.2 also splits Responses commentary and tool calls into alternative chat
choices, dropping tools for clients that consume only the first choice. Its
metadata-driven fake streaming also strips Codex's required wire stream flag.
The shims preserve model identifiers and credentials. Codex requests explicitly
select the Responses transport even when legacy model metadata lacks a new ID.
"""

from __future__ import annotations

import importlib
from functools import wraps
from importlib.metadata import version
from threading import RLock
from typing import Any

from pageindex import PageIndexClient

_install_lock = RLock()


class OAuthPageIndexClient(PageIndexClient):
    """Let LiteLLM authenticate ChatGPT via its real OAuth token cache."""

    @staticmethod
    def _validate_llm_provider(model: str) -> None:
        if model.startswith("chatgpt/"):
            return
        PageIndexClient._validate_llm_provider(model)


def coalesce_response_choices(choices: list[Any]) -> list[Any]:
    """Keep one Responses turn's commentary and tools in the same chat choice.

    Delegate parsing to upstream and merge only the known broken split shape.
    Already-correct upstream output and text-only responses remain untouched.
    Copy the first choice before merging so no upstream object is mutated.
    """
    if len(choices) < 2 or not any(choice.message.tool_calls for choice in choices):
        return choices
    merged = choices[0].model_copy(deep=True)
    messages = [choice.message for choice in choices]
    merged.message.content = (
        "\n".join(message.content for message in messages if message.content) or None
    )
    merged.message.tool_calls = [
        tool for message in messages for tool in (message.tool_calls or [])
    ]
    for field in ("annotations", "reasoning_items"):
        values = [item for message in messages for item in (getattr(message, field, None) or [])]
        if values:
            setattr(merged.message, field, values)
    reasoning = [getattr(message, "reasoning_content", None) for message in messages]
    if any(reasoning):
        merged.message.reasoning_content = "\n".join(value for value in reasoning if value)
    merged.finish_reason = "tool_calls"
    merged.index = 0
    return [merged]


def install_response_compat() -> None:
    """Install the stateless Responses repair once, without credential globals.

    The upstream conversion method has no model context. Its Responses output
    items represent a single turn, not alternative completions, for every
    provider. This pure structural fix is safe across concurrent KB requests.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    with _install_lock:
        if version("litellm") == "1.87.2":
            from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig

            original_stream = ChatGPTResponsesAPIConfig.should_fake_stream
            if not getattr(original_stream, "_openkb_codex_native_stream", False):

                @wraps(original_stream)
                def native_stream(self: Any, *args: Any, **kwargs: Any) -> bool:
                    # Codex requires SSE, including for unknown model IDs. The
                    # inherited OpenAI metadata fallback selects fake streaming;
                    # HTTPHandler then removes the provider's required stream=True.
                    # Nonstream callers keep upstream's buffered SSE parser.
                    return False

                setattr(native_stream, "_openkb_codex_native_stream", True)
                setattr(ChatGPTResponsesAPIConfig, "should_fake_stream", native_stream)

        # Legacy bundled metadata predates the requested GPT-6 IDs. Codex only
        # serves Responses; transport selection must not depend on that map.
        main = importlib.import_module("litellm.main")
        bridge = main.responses_api_bridge_check
        if not getattr(bridge, "_openkb_codex_responses", False):

            @wraps(bridge)
            def codex_bridge(model: str, custom_llm_provider: str, *args, **kwargs):
                if custom_llm_provider == "chatgpt":
                    return {"mode": "responses"}, model.removeprefix("responses/")
                return bridge(model, custom_llm_provider, *args, **kwargs)

            setattr(codex_bridge, "_openkb_codex_responses", True)
            setattr(main, "responses_api_bridge_check", codex_bridge)

        original = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices
        if getattr(original, "_openkb_coalesces_outputs", False):
            return

        @wraps(original)
        def convert(*args: Any, **kwargs: Any) -> list[Any]:
            return coalesce_response_choices(original(*args, **kwargs))

        convert._openkb_coalesces_outputs = True  # type: ignore[attr-defined]
        setattr(
            LiteLLMResponsesTransformationHandler,
            "_convert_response_output_to_choices",
            staticmethod(convert),
        )
