"""API settings: the generation config the API serves and the guardrail flags."""

from __future__ import annotations

from dataclasses import dataclass, field

from generation.config import GenerationConfig
from generation.guardrails import GuardrailConfig


@dataclass(frozen=True)
class ApiSettings:
    # The eval "full" config with the output guardrail on.
    generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(name="api", strict_claims=True)
    )
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
