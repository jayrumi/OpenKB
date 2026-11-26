"""Request-local model selection for the three knowledge-base pipeline stages."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Literal

from openkb.config import DEFAULT_CONFIG

Stage = Literal["indexing", "compilation", "query"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def require_chat_model_allowed(model: str) -> None:
    """Apply an opt-in deployment allowlist without changing stored models.

    OPENKB_CHAT_ALLOWED_MODELS is a JSON list of exact provider/model tokens.
    When absent, legacy deployments retain their existing chat behavior.
    This is chat admission, not a provider policy for indexing/compilation.
    """
    raw = os.environ.get("OPENKB_CHAT_ALLOWED_MODELS")
    if raw is None:
        return
    error = "OPENKB_CHAT_ALLOWED_MODELS must be a JSON list of non-empty model strings"
    try:
        allowed = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(error) from None
    if not isinstance(allowed, list) or any(
        not isinstance(token, str) or not token.strip() for token in allowed
    ):
        raise ValueError(error)
    if model not in allowed:
        raise ValueError(
            f"Chat model {model!r} is not allowed by this deployment. "
            "Please start a new chat with the configured query model; "
            "your saved history is unchanged."
        )


def resolve_stage_reasoning(config: Mapping[str, Any], stage: Stage) -> str | None:
    """Null/absent stage effort inherits shared effort; both unset omit it.

    Effective-config layering first applies KB null -> global inheritance.
    No model token, credential, process global, or caller mapping is modified.
    """
    if stage not in ("indexing", "compilation", "query"):
        raise ValueError(f"Unknown reasoning stage: {stage!r}")
    key = f"{stage}_reasoning_effort"
    value = config.get(key)
    if value is None:
        key, value = "reasoning_effort", config.get("reasoning_effort")
    if value is not None and (not isinstance(value, str) or value not in REASONING_EFFORTS):
        raise ValueError(f"{key} must be a supported reasoning effort or null")
    return value


def resolve_stage_model(config: Mapping[str, Any], stage: Stage) -> str:
    """Return a stage override, otherwise the unchanged legacy model fallback.

    Missing and null stage overrides inherit ``model``. Blank or non-string
    overrides are configuration errors. Provider/model tokens are returned
    verbatim: selecting a model does not validate provider support or rewrite
    identifiers. Nothing is cached or written to config or process globals.
    """
    if stage not in ("indexing", "compilation", "query"):
        raise ValueError(f"Unknown model stage: {stage!r}")
    key = f"{stage}_model"
    value = config.get(key)
    if value is None:
        return config.get("model", DEFAULT_CONFIG["model"])
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty model string or null")
    return value
