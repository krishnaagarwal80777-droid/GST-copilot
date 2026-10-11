from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from gst_copilot.models import ProvisionRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def bm25_rank(
    query: str, records: Sequence[ProvisionRecord], *, k1: float = 1.5, b: float = 0.75
) -> list[tuple[ProvisionRecord, float]]:
    """Simplified BM25 ranking using token overlap for a baseline retrieval layer."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    docs = []
    for record in records:
        text = " ".join(filter(None, [record.heading or "", record.text or "", record.path]))
        tokens = _tokenize(text)
        docs.append((record, tokens))

    N = len(docs)
    if N == 0:
        return []

    doc_lengths = [len(tokens) for _, tokens in docs]
    avgdl = sum(doc_lengths) / N
    df = Counter()
    for _, tokens in docs:
        df.update(set(tokens))

    ranked: list[tuple[ProvisionRecord, float]] = []
    for record, tokens in docs:
        tf = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(((N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5)) + 1.0)
            numerator = tf[token] * (k1 + 1.0)
            denominator = tf[token] + k1 * (1.0 - b + b * (len(tokens) / avgdl))
            score += idf * (numerator / denominator)
        if score > 0:
            ranked.append((record, score))

    ranked.sort(key=lambda item: (-item[1], item[0].path))
    return ranked


class Bm25Index:
    """bm25s index over row texts; scores every row so a date mask can be applied first."""

    def __init__(self, texts: Sequence[str]) -> None:
        import bm25s

        self._bm25s = bm25s
        self.retriever = bm25s.BM25()
        self.retriever.index(
            bm25s.tokenize(list(texts), stopwords="en", show_progress=False), show_progress=False
        )
        self.size = len(texts)

    def rank(self, query: str, *, mask: Sequence[bool] | None = None, k: int = 50) -> list[int]:
        """Row indices by descending BM25 score (positive scores only), optionally masked."""
        tokens = self._bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.ids or not tokens.ids[0]:
            return []
        # Query ids index the query's own vocab; get_scores takes the words themselves.
        id_to_word = {token_id: word for word, token_id in tokens.vocab.items()}
        words = [id_to_word[token_id] for token_id in tokens.ids[0]]
        scores = self.retriever.get_scores(words)
        order = sorted(range(self.size), key=lambda index: -scores[index])
        return [index for index in order if scores[index] > 0 and (mask is None or mask[index])][:k]
