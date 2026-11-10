"""POST /ask without HTTP: AskRequest -> AskResponse through generation.answer.Answerer.

Input guardrails (generation.guardrails) run first; a refused question gets the guardrail's
fixed message and never reaches the generator. The output guardrail is the Answerer's
strict_claims flag (on in api.config.ApiSettings). Every request writes one line to the
request log (observability.log).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

from api.config import ApiSettings
from api.schemas import AskRequest, AskResponse, Citation, ClaimOut
from api.sources import source_file, source_url
from generation.answer import Answerer, AnswerResult
from generation.context import Chunk
from generation.guardrails import (
    Guardrail,
    check_after_retrieval,
    check_before_retrieval,
    refusal,
)
from generation.prompts import chunk_label
from generation.timing import ms
from observability.log import RequestLogger, llm_fields, now, question_hash
from retrieval.store import OPEN_END

log = logging.getLogger("gst_copilot.api")


def citations_for(
    result: AnswerResult, chunks: list[Chunk]
) -> tuple[list[ClaimOut], list[Citation]]:
    """Number the cited chunks in order of first citation."""
    by_id = {chunk.id: chunk for chunk in chunks}
    numbers: dict[str, int] = {}
    citations: list[Citation] = []
    claims: list[ClaimOut] = []
    for claim in result.claims:
        for cid in claim["citation_ids"]:
            if cid in numbers or cid not in by_id:
                continue
            chunk = by_id[cid]
            numbers[cid] = len(numbers) + 1
            citations.append(
                Citation(
                    number=numbers[cid],
                    id=cid,
                    version_id=chunk.record_id,
                    path=chunk.path,
                    label=chunk_label(chunk.path),
                    doc_type=chunk.doc_type,
                    effective_from=date.fromisoformat(chunk.effective_from),
                    effective_to=(
                        None
                        if chunk.effective_to == OPEN_END
                        else date.fromisoformat(chunk.effective_to)
                    ),
                    source_file=source_file(chunk.record_id),
                    source_url=source_url(chunk.record_id),
                    text=chunk.text,
                )
            )
        claims.append(
            ClaimOut(
                text=claim["text"],
                kind=claim["kind"],
                citation_ids=claim["citation_ids"],
                citation_numbers=[numbers[cid] for cid in claim["citation_ids"] if cid in numbers],
            )
        )
    return claims, citations


class AskService:
    def __init__(
        self,
        answerer: Answerer | Callable[[], Answerer],
        settings: ApiSettings | None = None,
        logger: RequestLogger | None = None,
    ) -> None:
        # A callable is called on first use (loading the models takes a while).
        self._answerer = answerer if isinstance(answerer, Answerer) else None
        self._factory = None if isinstance(answerer, Answerer) else answerer
        self.settings = settings or ApiSettings()
        self.logger = logger

    @property
    def loaded(self) -> bool:
        return self._answerer is not None

    @property
    def answerer(self) -> Answerer:
        if self._answerer is None:
            self._answerer = self._factory()
        return self._answerer

    def ask(self, request: AskRequest) -> AskResponse:
        """Answer one request and write its log line (also when the pipeline raises)."""
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "timestamp": now(),
            "question_sha256": question_hash(request.question),
            "question_chars": len(request.question),
            "reference_date": request.reference_date.isoformat(),
            "knowledge_date": request.knowledge_date and request.knowledge_date.isoformat(),
            "timings_ms": {},
            "retrieved_ids": [],
            "gate_score": None,
            "abstained": None,
            "abstain_reason": None,
            "guardrail": None,
            **llm_fields(None),
            "status": "error",
        }
        try:
            response = self._ask(request, trace, started)
            trace["status"] = "ok"
            return response
        finally:
            trace["timings_ms"]["total"] = ms(time.perf_counter() - started)
            if self.logger is not None:
                self.logger.write(trace)

    def _ask(self, request: AskRequest, trace: dict[str, Any], started: float) -> AskResponse:
        guards = self.settings.guardrails
        tripped = check_before_retrieval(request.question, guards)
        if tripped:
            return self._refuse(request, trace, tripped, started)
        answerer = self.answerer
        context, timings = answerer.retrieve(
            request.question, request.reference_date, known_by=request.knowledge_date
        )
        trace["timings_ms"].update(timings)
        trace["retrieved_ids"] = [chunk.id for chunk in context.chunks]
        trace["gate_score"] = round(context.confidence, 4)
        tripped = check_after_retrieval(request.question, context.confidence, guards)
        if tripped:
            return self._refuse(request, trace, tripped, started)
        result = answerer.answer(
            request.question,
            request.reference_date,
            known_by=request.knowledge_date,
            context=context,
        )
        trace["timings_ms"].update(result.timings_ms)
        trace.update(
            abstained=result.abstained,
            abstain_reason=result.abstain_reason,
            **llm_fields(result.llm),
        )
        claims, citations = citations_for(result, context.chunks)
        return AskResponse(
            request_id=trace["request_id"],
            answer=result.rendered,
            claims=claims,
            citations=citations,
            assumptions=result.assumptions,
            abstained=result.abstained,
            abstain_reason=result.abstain_reason,
            guardrail=None,
            reference_date=request.reference_date,
            knowledge_date=request.knowledge_date,
            timings_ms={**trace["timings_ms"], "total": ms(time.perf_counter() - started)},
            model=trace["model"],
        )

    def _refuse(
        self, request: AskRequest, trace: dict[str, Any], guardrail: Guardrail, started: float
    ) -> AskResponse:
        if guardrail == "injection":
            log.warning(
                "prompt-injection guardrail: request %s, question sha256 %s",
                trace["request_id"],
                trace["question_sha256"],
            )
        trace.update(abstained=True, abstain_reason="guardrail", guardrail=guardrail)
        return AskResponse(
            request_id=trace["request_id"],
            answer=refusal(guardrail, self.settings.guardrails),
            claims=[],
            citations=[],
            assumptions=[],
            abstained=True,
            abstain_reason="guardrail",
            guardrail=guardrail,
            reference_date=request.reference_date,
            knowledge_date=request.knowledge_date,
            timings_ms={**trace["timings_ms"], "total": ms(time.perf_counter() - started)},
            model=None,
        )
