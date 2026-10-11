"""Local bge-m3 dense embeddings, cached to disk (no hosted embedding API).

Each text's vector is stored under data/cache/embeddings/<model>/<sha256(text)>.npy, so
re-indexing and re-running evals only embed texts that changed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "BAAI/bge-m3"
CACHE_DIR = ROOT / "data" / "cache" / "embeddings" / "bge-m3"
MAX_SEQ_LENGTH = 1024
DIMENSION = 1024


class Embedder:
    def __init__(self, cache_dir: Path = CACHE_DIR, batch_size: int = 16) -> None:
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Cached weights only: no network, and no background fetch of an alternate
                # weights format on every load.
                model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
            except OSError:
                model = SentenceTransformer(MODEL_NAME, device=device)
            model.max_seq_length = MAX_SEQ_LENGTH
            if device == "cuda":
                model = model.half()
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        return self._load().tokenizer

    def _path(self, text: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.npy"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-normalised float32 vectors, one row per text."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        vectors: list[np.ndarray | None] = []
        missing: list[int] = []
        for index, text in enumerate(texts):
            path = self._path(text)
            if path.is_file():
                vectors.append(np.load(path))
            else:
                vectors.append(None)
                missing.append(index)
        if missing:
            encoded = (
                self._load()
                .encode(
                    [texts[index] for index in missing],
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=len(missing) > 64,
                )
                .astype(np.float32)
            )
            for index, vector in zip(missing, encoded, strict=True):
                np.save(self._path(texts[index]), vector)
                vectors[index] = vector
        return np.vstack(vectors) if vectors else np.zeros((0, DIMENSION), dtype=np.float32)
