"""Question + reference date -> cited answer or abstention.

1. Context: top servable chunks valid on the reference date (generation.context).
2. Gate: abstain without calling the LLM when no servable chunk reaches the threshold.
3. Date gate: abstain without calling the LLM when the reference date is after the knowledge
   horizon of the question's source type (generation.horizon). The source type is that of the
   leading context chunk: the pinned exact-match provision if the question cites one,
   otherwise the best-scoring chunk. It runs after the confidence gate so that the source
   type is read from a context that matches the question.
4. Generate: JSON claims with citation ids (generation.schemas), temperature 0, cached.
5. Verify: drop citation ids not in the context, then claims left without one. With
   ``strict_claims`` (output guardrail) the output must have every schema field, and any
   claim without a context citation abstains instead.
6. Render the fixed template, or the abstention template with the closest provisions.

With ``tools`` (calculator routing, generation.tools_router; on in the API, off in the eval
configs) a planner turn runs first. When it calls a tool, the tool outputs become TOOL-<n>
context chunks and replace the gates: the retrieved provisions are added only when the gates
pass. A refused, ambiguous or failed final tool call abstains. When a tool ran, every number
in the answer's claims and assumptions must appear in a tool output, a cited chunk or the
question, otherwise the answer abstains (``unsupported_number``). Answers that used no tool
are checked as before (citations, and strict_claims when on), not for numbers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import ValidationError

from generation.config import GenerationConfig
from generation.context import Chunk, Context, build_context
from generation.horizon import SOURCE_LABELS, knowledge_horizons
from generation.llm import LLMClient
from generation.prompts import generator_prompt, generator_system
from generation.render import render_abstention, render_answer
from generation.schemas import GenerationOutput, StrictGenerationOutput, response_schema
from generation.timing import TimedReranker, ms
from generation.tools_router import (
    TOOL_ANSWER_RULES,
    ToolCall,
    ToolRun,
    plan_and_run,
    provisions_by_path,
    tool_chunks,
    unsupported_numbers,
)
from generation.verify import Verified, verify
from retrieval.pipeline import Retriever

AbstainReason = Literal[
    "horizon",
    "gate",
    "model",
    "no_supported_claims",
    "unsupported_claim",
    "invalid_output",
    "ambiguous_lookup",
    "not_found",
    "tool_error",
    "unsupported_number",
]
ABSTAIN_TEXT: dict[str, str] = {
    "horizon": (
        "The reference date {ref_date} is after {horizon}, the last date the corpus covers "
        "for {source}, so the law on that date cannot be confirmed."
    ),
    "gate": "No provision valid on the reference date matches the question closely enough.",
    "model": "The provisions valid on the reference date do not answer the question.",
    "no_supported_claims": "No statement in the draft answer was supported by a cited provision.",
    "unsupported_claim": (
        "A statement in the draft answer did not cite a retrieved provision, so no answer is given."
    ),
    "invalid_output": "The answer could not be produced in the required structured form.",
    "ambiguous_lookup": "More than one rate-schedule entry matches, so no rate is given.",
    "not_found": "No rate-schedule entry in force on the reference date matches.",
    "tool_error": "A calculator call failed, so no answer is given.",
    "unsupported_number": (
        "The draft answer contained a number that is not in the calculator output or the cited "
        "provisions ({numbers}), so no answer is given."
    ),
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
    llm: dict[str, Any] | None = None  # provider, model, cached, usage, latency_s
    prompt_tokens: dict[str, Any] | None = None  # dry run only
    # Stage -> milliseconds (generation.timing.STAGES); retrieve and rerank are filled only
    # when this call built the context.
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Calculator routing only (GenerationConfig.tools): the tool calls made (ToolCall.to_dict)
    # and one llm entry per planner turn. Left out of to_dict when empty, so eval records
    # without tools are unchanged.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    plan_llm: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("tool_calls", "plan_llm"):
            if not data[key]:
                del data[key]
        return data

    @property
    def cited_paths(self) -> list[str]:
        """Provision paths cited by the verified claims (chunk id -> path)."""
        paths = (cid.split("@", 1)[0] for claim in self.claims for cid in claim["citation_ids"])
        return list(dict.fromkeys(paths))


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        generator: LLMClient,
        config: GenerationConfig | None = None,
        *,
        horizons: dict[str, date] | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config or GenerationConfig()
        self._horizons = horizons
        self._provisions: dict[str, list[dict]] | None = None
        if retriever is not None and retriever.reranker is not None:
            if not isinstance(retriever.reranker, TimedReranker):
                retriever.reranker = TimedReranker(retriever.reranker)

    @property
    def horizons(self) -> dict[str, date]:
        """doc_type -> knowledge horizon; read from the sources on first use."""
        if self._horizons is None:
            self._horizons = knowledge_horizons()
        return self._horizons

    def prompt(self, question: str, ref_date: date, chunks: list[Chunk]) -> str:
        return generator_prompt(question, ref_date, chunks)

    @property
    def provisions(self) -> dict[str, list[dict]] | None:
        """Index rows by path (for gst_split's citations); None without a row table."""
        if self._provisions is None and hasattr(self.retriever, "_table"):
            from retrieval.pipeline import PROVISIONS_TABLE

            self._provisions = provisions_by_path(self.retriever._table(PROVISIONS_TABLE))
        return self._provisions

    def plan_tools(self, question: str, ref_date: date) -> ToolRun:
        return plan_and_run(self.generator, question, ref_date, self.provisions)

    def _gate_reason(
        self, chunks: list[Chunk], context: Context, ref_date: date
    ) -> tuple[AbstainReason, str | None] | None:
        """Why the confidence gate or the date gate stops this context, or None."""
        if not chunks or (self.config.gate and context.confidence < self.config.abstain_threshold):
            return "gate", None
        if self.config.date_gate:
            source = chunks[0].doc_type
            horizon = self.horizons.get(source)
            if horizon is not None and ref_date > horizon:
                text = ABSTAIN_TEXT["horizon"].format(
                    ref_date=ref_date.isoformat(),
                    horizon=horizon.isoformat(),
                    source=SOURCE_LABELS[source],
                )
                return "horizon", text
        return None

    def retrieve(
        self, question: str, ref_date: date, *, known_by: date | None = None
    ) -> tuple[Context, dict[str, float]]:
        """The generator context and its retrieve/rerank timings (ms); no LLM call."""
        reranker = getattr(self.retriever, "reranker", None)
        timed = reranker if isinstance(reranker, TimedReranker) else None
        if timed is not None:
            timed.reset()
        started = time.perf_counter()
        context = build_context(self.retriever, question, ref_date, self.config, known_by=known_by)
        total = time.perf_counter() - started
        rerank = timed.elapsed() if timed is not None else 0.0
        return context, {"retrieve": ms(total - rerank), "rerank": ms(rerank)}

    def answer(
        self,
        question: str,
        ref_date: date,
        *,
        known_by: date | None = None,
        dry_run: bool = False,
        context: Context | None = None,
    ) -> AnswerResult:
        """Answer from ``context`` when given (``retrieve`` with the same arguments),
        otherwise build it."""
        timings: dict[str, float] = {}
        if context is None:
            context, timings = self.retrieve(question, ref_date, known_by=known_by)
        chunks = context.chunks
        base = {
            "question": question,
            "ref_date": ref_date.isoformat(),
            "context_ids": [chunk.id for chunk in chunks],
            "confidence": round(context.confidence, 4),
            "timings_ms": timings,
        }

        tools: dict[str, Any] = {}  # tool_calls and plan_llm, when calculator routing ran

        def abstain(reason: AbstainReason, text: str | None = None, **extra: Any) -> AnswerResult:
            rendered = render_abstention(text or ABSTAIN_TEXT[reason], ref_date, chunks)
            return AnswerResult(
                **base, abstained=True, abstain_reason=reason, rendered=rendered, **extra, **tools
            )

        used_tools = False
        if self.config.tools and not dry_run:
            run = self.plan_tools(question, ref_date)
            timings.update(plan=run.plan_ms, tools=run.tools_ms)
            tools = {"tool_calls": [call.to_dict() for call in run.calls], "plan_llm": run.llm}
            used_tools = bool(run.calls)
        if used_tools:
            problem = tool_problem(run.calls)
            if problem:
                return abstain(*problem)
            # The retrieved provisions join the calculator outputs only when the gates pass.
            law = chunks if self._gate_reason(chunks, context, ref_date) is None else []
            chunks = tool_chunks(run.calls, ref_date) + law
        else:
            stop = self._gate_reason(chunks, context, ref_date)
            if stop:
                return abstain(*stop)
        prompt = self.prompt(question, ref_date, chunks)
        system = generator_system(self.config.prompt_variant)
        if used_tools:
            system += TOOL_ANSWER_RULES
        schema = response_schema(self.config.prompt_variant)
        if dry_run:
            count = self.generator.count_tokens(system, prompt)
            return AnswerResult(
                **base,
                abstained=False,
                abstain_reason=None,
                rendered="",
                prompt_tokens={
                    "input_tokens": count.input_tokens,
                    "method": count.method,
                    "cached": self.generator.is_cached(system, prompt, schema=schema),
                },
            )
        started = time.perf_counter()
        response = self.generator.complete(system, prompt, schema=schema)
        timings["generate"] = ms(time.perf_counter() - started)
        started = time.perf_counter()
        llm = {
            "provider": response.provider,
            "model": response.model,
            "cached": response.cached,
            "usage": response.usage,
            "latency_s": response.latency_s,
        }
        try:
            schema_model = StrictGenerationOutput if self.config.strict_claims else GenerationOutput
            output = schema_model.model_validate_json(response.text)
        except ValidationError:
            timings["verify"] = ms(time.perf_counter() - started)
            return abstain("invalid_output", raw_output=response.text, llm=llm)
        if output.abstain:
            timings["verify"] = ms(time.perf_counter() - started)
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
        if self.config.strict_claims:
            allowed = {chunk.id for chunk in chunks}  # the retrieved context, plus any TOOL- chunks
            if checked.dropped_claims or any(
                not allowed & set(claim.citation_ids) for claim in checked.claims
            ):
                timings["verify"] = ms(time.perf_counter() - started)
                return abstain("unsupported_claim", **dumped)
        if not checked.claims:
            timings["verify"] = ms(time.perf_counter() - started)
            return abstain("no_supported_claims", **dumped)
        if used_tools:
            cited = {cid for claim in checked.claims for cid in claim.citation_ids}
            sources = [c.text for c in chunks if c.id in cited or c.doc_type == "tool"]
            written = [claim.text for claim in checked.claims] + list(output.assumptions)
            missing = unsupported_numbers(written, [*sources, question])
            if missing:
                timings["verify"] = ms(time.perf_counter() - started)
                text = ABSTAIN_TEXT["unsupported_number"].format(numbers=", ".join(missing))
                return abstain("unsupported_number", text, **dumped)
        # Rendering states the reference date the pipeline used, whatever the model wrote.
        rendered = render_answer(checked.claims, output.assumptions, ref_date, chunks)
        timings["verify"] = ms(time.perf_counter() - started)
        return AnswerResult(
            **base,
            abstained=False,
            abstain_reason=None,
            rendered=rendered,
            claims=[claim.model_dump() for claim in checked.claims],
            assumptions=output.assumptions,
            **dumped,
            **tools,
        )


def tool_problem(calls: list[ToolCall]) -> tuple[AbstainReason, str] | None:
    """Abstention for the last call of each tool when it failed, was refused or found no
    single row (an earlier ambiguous lookup that the planner then narrowed does not count)."""
    last = {call.name: call for call in calls}
    for call in last.values():
        if not call.ok:
            return "tool_error", f"{ABSTAIN_TEXT['tool_error'][:-1]} ({call.name}: {call.error})."
        status = call.output["status"]
        if status == "refused":
            return "horizon", " ".join(call.output["notes"])
        if status == "ambiguous":
            options = "; ".join(
                f"Notification {c['notification']} Schedule {c['schedule']} S. No. {c['s_no']} "
                f"({c['tariff']} {c['description'][:80]}): central tax {c['rate']}"
                for c in call.output["candidates"]
            )
            return "ambiguous_lookup", f"{ABSTAIN_TEXT['ambiguous_lookup']} Candidates: {options}."
        if status == "not_found":
            return "not_found", ABSTAIN_TEXT["not_found"]
    return None
