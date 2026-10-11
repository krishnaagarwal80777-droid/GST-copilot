"""Question + reference date -> cited answer or abstention.

1. Context: top servable chunks valid on the reference date (generation.context).
2. Gate: abstain without calling the LLM when no servable chunk reaches the threshold.
3. Generate: JSON claims with citation ids (generation.schemas), temperature 0, cached.
4. Verify: drop citation ids not in the context, then claims left without one.
5. Render the fixed template, or the abstention template with the closest provisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import ValidationError

from generation.config import GenerationConfig
from generation.context import Chunk, build_context
from generation.llm import LLMClient
from generation.prompts import GENERATOR_SYSTEM, generator_prompt
from generation.render import render_abstention, render_answer
from generation.schemas import RESPONSE_SCHEMA, GenerationOutput
from generation.verify import Verified, verify
from retrieval.pipeline import Retriever

AbstainReason = Literal["gate", "model", "no_supported_claims", "invalid_output"]
ABSTAIN_TEXT: dict[str, str] = {
    "gate": "No provision valid on the reference date matches the question closely enough.",
    "model": "The provisions valid on the reference date do not answer the question.",
    "no_supported_claims": "No statement in the draft answer was supported by a cited provision.",
    "invalid_output": "The answer could not be produced in the required structured form.",
}


@dataclass
class AnswerResult:
    question: str
    ref_date: str
    context_ids: list[str]
    confidence: float
    abstained: bool
    abstain_reason: AbstainReason | None
    rendered: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    dropped_claims: list[dict[str, Any]] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ref_date_matches: bool | None = None
    raw_output: str | None = None
    llm: dict[str, Any] | None = None  # provider, model, cached, usage
    prompt_tokens: dict[str, Any] | None = None  # dry run only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cited_paths(self) -> list[str]:
        """Provision paths cited by the verified claims (chunk id -> path)."""
        paths = (cid.split("@", 1)[0] for claim in self.claims for cid in claim["citation_ids"])
        return list(dict.fromkeys(paths))


class Answerer:
    def __init__(
        self, retriever: Retriever, generator: LLMClient, config: GenerationConfig | None = None
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config or GenerationConfig()

    def prompt(self, question: str, ref_date: date, chunks: list[Chunk]) -> str:
        return generator_prompt(question, ref_date, chunks)

    def answer(
        self,
        question: str,
        ref_date: date,
        *,
        known_by: date | None = None,
        dry_run: bool = False,
    ) -> AnswerResult:
        context = build_context(self.retriever, question, ref_date, self.config, known_by=known_by)
        chunks = context.chunks
        base = {
            "question": question,
            "ref_date": ref_date.isoformat(),
            "context_ids": [chunk.id for chunk in chunks],
            "confidence": round(context.confidence, 4),
        }

        def abstain(reason: AbstainReason, **extra: Any) -> AnswerResult:
            rendered = render_abstention(ABSTAIN_TEXT[reason], ref_date, chunks)
            return AnswerResult(
                **base, abstained=True, abstain_reason=reason, rendered=rendered, **extra
            )

        if not chunks or (self.config.gate and context.confidence < self.config.abstain_threshold):
            return abstain("gate")
        prompt = self.prompt(question, ref_date, chunks)
        if dry_run:
            count = self.generator.count_tokens(GENERATOR_SYSTEM, prompt)
            return AnswerResult(
                **base,
                abstained=False,
                abstain_reason=None,
                rendered="",
                prompt_tokens={
                    "input_tokens": count.input_tokens,
                    "method": count.method,
                    "cached": self.generator.is_cached(
                        GENERATOR_SYSTEM, prompt, schema=RESPONSE_SCHEMA
                    ),
                },
            )
        response = self.generator.complete(GENERATOR_SYSTEM, prompt, schema=RESPONSE_SCHEMA)
        llm = {
            "provider": response.provider,
            "model": response.model,
            "cached": response.cached,
            "usage": response.usage,
        }
        try:
            output = GenerationOutput.model_validate_json(response.text)
        except ValidationError:
            return abstain("invalid_output", raw_output=response.text, llm=llm)
        if output.abstain:
            return abstain("model", raw_output=response.text, llm=llm)
        if self.config.verify_citations:
            checked = verify(output, chunks, ref_date.isoformat())
        else:  # ablation: keep every claim as generated
            matches = output.ref_date_used.strip() == base["ref_date"]
            checked = Verified(output.claims, ref_date_matches=matches)
        dumped = {
            "dropped_claims": [claim.model_dump() for claim in checked.dropped_claims],
            "dropped_citations": checked.dropped_citations,
            "ref_date_matches": checked.ref_date_matches,
            "raw_output": response.text,
            "llm": llm,
        }
        if not checked.claims:
            return abstain("no_supported_claims", **dumped)
        # Rendering states the reference date the pipeline used, whatever the model wrote.
        rendered = render_answer(checked.claims, output.assumptions, ref_date, chunks)
        return AnswerResult(
            **base,
            abstained=False,
            abstain_reason=None,
            rendered=rendered,
            claims=[claim.model_dump() for claim in checked.claims],
            assumptions=output.assumptions,
            **dumped,
        )
