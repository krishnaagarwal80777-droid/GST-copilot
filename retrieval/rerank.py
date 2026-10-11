"""bge-reranker-v2-m3 cross-encoder (local weights, GPU when available)."""

from __future__ import annotations

from collections.abc import Sequence

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512


class Reranker:
    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {"device": device, "max_length": MAX_LENGTH}
            try:
                model = CrossEncoder(RERANK_MODEL, local_files_only=True, **kwargs)
            except OSError:
                model = CrossEncoder(RERANK_MODEL, **kwargs)
            if device == "cuda":
                model.model.half()
            self._model = model
        return self._model

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._load().predict(pairs, batch_size=self.batch_size)]
