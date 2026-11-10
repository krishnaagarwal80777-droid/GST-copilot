"""FastAPI app: POST /ask, GET /health.

    uvicorn api.app:app --port 8000

The retrieval models and the generator are loaded on the first /ask, not at import.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.config import ApiSettings
from api.schemas import AskRequest, AskResponse, Health
from api.service import AskService
from generation.answer import Answerer
from generation.llm import LLMError
from observability.log import RequestLogger


def default_answerer(settings: ApiSettings) -> Answerer:
    from generation.llm import generator_client
    from retrieval.embed import Embedder
    from retrieval.pipeline import Retriever
    from retrieval.rerank import Reranker
    from retrieval.store import open_store

    retriever = Retriever(open_store(), Embedder(), Reranker())
    return Answerer(retriever, generator_client(), settings.generation)


def create_app(service: AskService | None = None) -> FastAPI:
    if service is None:
        settings = ApiSettings()
        service = AskService(lambda: default_answerer(settings), settings, RequestLogger())
    app = FastAPI(title="GST Tax Copilot", version="0.1.0")
    app.state.service = service

    @app.get("/health", response_model=Health)
    def health() -> Health:
        store = service.answerer.retriever.store.backend if service.loaded else None
        return Health(status="ok", pipeline_loaded=service.loaded, store=store)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            return service.ask(request)
        except LLMError as error:
            raise HTTPException(503, f"generator unavailable: {type(error).__name__}") from error

    return app


app = create_app()
