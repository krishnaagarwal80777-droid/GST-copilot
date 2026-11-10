"""POST /ask without HTTP: AskRequest -> AskResponse through generation.answer.Answerer.

Input guardrails (generation.guardrails) run first; a refused question gets the guardrail's
fixed message and never reaches the generator. The output guardrail is the Answerer's
strict_claims flag (on in api.config.ApiSettings).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from datetime import date

from api.config import ApiSettings
from api.schemas import AskRequest, AskResponse, Citation, ClaimOut
from api.sources import source_url
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


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


class AskService:
    def __init__(
        self,
        answerer: Answerer | Callable[[], Answerer],
        settings: ApiSettings | None = None,
    ) -> None:
        # A callable is called on first use (loading the models takes a while).
        self._answerer = answerer if isinstance(answerer, Answerer) else None
        self._factory = None if isinstance(answerer, Answerer) else answerer
        self.settings = settings or ApiSettings()

    @property
    def loaded(self) -> bool:
        return self._answerer is not None

    @property
    def answerer(self) -> Answerer:
        if self._answerer is None:
            self._answerer = self._factory()
        return self._answerer

    def _refuse(
        self,
        request: AskRequest,
        request_id: str,
        guardrail: Guardrail,
        timings: dict[str, float],
        started: float,
    ) -> AskResponse:
        if guardrail == "injection":
            log.warning(
                "prompt-injection guardrail: request %s, question sha256 %s",
                request_id,
                question_hash(request.question),
            )
        return AskResponse(
            request_id=request_id,
            answer=refusal(guardrail, self.settings.guardrails),
            claims=[],
            citations=[],
            abstained=True,
            abstain_reason="guardrail",
            guardrail=guardrail,
            reference_date=request.reference_date,
            knowledge_date=request.knowledge_date,
            timings_ms={**timings, "total": ms(time.perf_counter() - started)},
            model=None,
        )

    def ask(self, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        guards = self.settings.guardrails
        tripped = check_before_retrieval(request.question, guards)
        if tripped:
            return self._refuse(request, request_id, tripped, {}, started)
        answerer = self.answerer
        context, timings = answerer.retrieve(
            request.question, request.reference_date, known_by=request.knowledge_date
        )
        tripped = check_after_retrieval(request.question, context.confidence, guards)
        if tripped:
            return self._refuse(request, request_id, tripped, timings, started)
        result = answerer.answer(
            request.question,
            request.reference_date,
            known_by=request.knowledge_date,
            context=context,
        )
        timings = {**timings, **result.timings_ms, "total": ms(time.perf_counter() - started)}
        claims, citations = citations_for(result, context.chunks)
        return AskResponse(
            request_id=request_id,
            answer=result.rendered,
            claims=claims,
            citations=citations,
            abstained=result.abstained,
            abstain_reason=result.abstain_reason,
            guardrail=None,
            reference_date=request.reference_date,
            knowledge_date=request.knowledge_date,
            timings_ms=timings,
            model=(result.llm or {}).get("model"),
        )
