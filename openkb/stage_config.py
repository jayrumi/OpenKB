"""Request-local model selection for the three knowledge-base pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from openkb.config import DEFAULT_CONFIG

Stage = Literal["indexing", "compilation", "query"]


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
