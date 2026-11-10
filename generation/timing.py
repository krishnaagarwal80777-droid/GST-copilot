"""Stage timings for one answer: retrieve, rerank, generate, verify (milliseconds).

The cross-encoder runs inside retrieval (re-ranking the fused head) and again when the context
is built, so its time cannot be read off one call. ``TimedReranker`` wraps the reranker and
adds up the time spent in ``scores`` per thread; ``retrieve`` is the context build time minus
that.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

STAGES = ("retrieve", "rerank", "generate", "verify")


class TimedReranker:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._local = threading.local()

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        started = time.perf_counter()
        try:
            return self.inner.scores(query, texts)
        finally:
            self._local.seconds = self.elapsed() + time.perf_counter() - started

    def elapsed(self) -> float:
        return getattr(self._local, "seconds", 0.0)

    def reset(self) -> None:
        self._local.seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
