from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pytest

from retrieval.config import CONFIGS, RetrievalConfig, ablation_configs
from retrieval.exact_match import exact_paths, parse_citations
from retrieval.fusion import rrf

pytest.importorskip("lancedb")
pytest.importorskip("bm25s")

from retrieval.pipeline import PROVISIONS_TABLE, Retriever  # noqa: E402
from retrieval.store import DEFAULT_PG_DSN, OPEN_END, LanceStore, PgVectorStore  # noqa: E402


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Is credit blocked under 17(5)?", ["CGST/s17/ss5"]),
        ("What is the limit in s.16(4)?", ["CGST/s16/ss4"]),
        ("section 17(5)(a) of the CGST Act", ["CGST/s17/ss5/cl(a)"]),
        ("What does rule 36(4) say?", ["CGST/r36/ss4"]),
        ("rate under Notification 9/2025-CT(Rate)", ["CTR/9-2025"]),
        ("Notification No. 1/2017-Central Tax (Rate)", ["CTR/1-2017"]),
        ("condition in section 16(2)(aa)", ["CGST/s16/ss2/cl(aa)"]),
        ("invoices of financial year 2017-18 and a bus of 20 seats", []),
    ],
)
def test_citations_normalise_to_canonical_paths(query, expected) -> None:
    assert parse_citations(query) == expected


def test_exact_paths_prefers_the_cited_node_then_children_then_ancestor() -> None:
    indexed = ["CGST/s16/ss4", "CGST/s16/ss4/proviso1", "CGST/r89"]
    assert exact_paths("CGST/s16/ss4", indexed) == ["CGST/s16/ss4", "CGST/s16/ss4/proviso1"]
    assert exact_paths("CGST/r89/ss1", indexed) == ["CGST/r89"]
    assert exact_paths("CGST/s77", indexed) == []


def test_rrf_rewards_agreement() -> None:
    # Equal fused scores tie-break by id; an item in both lists beats one in a single list.
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"]
    assert rrf([["x", "y"], ["y", "z"]])[0] == "y"


def test_ablation_ladder_is_cumulative() -> None:
    names = [config.name for config in CONFIGS]
    assert names == [
        "naive",
        "dense",
        "bm25",
        "dense+bm25",
        "+exact_match",
        "+date_filter",
        "+reranker",
    ]
    full = CONFIGS[-2]
    assert (full.dense, full.bm25, full.exact_match, full.date_filter) == (True, True, True, True)
    assert CONFIGS[-1].rerank and not full.rerank
    # split_bm25 is on for every rung that uses BM25, and can be switched off for the ladder.
    assert [config.split_bm25 for config in CONFIGS] == [config.bm25 for config in CONFIGS]
    assert not any(config.split_bm25 for config in ablation_configs(split_bm25=False))


def test_weighted_rrf_and_dense_first() -> None:
    from retrieval.fusion import dense_first, weighted_rrf

    dense, bm25 = ["a", "b", "c"], ["c", "b", "d"]
    # Equal weights reward agreement; a zero BM25 weight reduces to the dense order.
    assert weighted_rrf([dense, bm25], [1.0, 1.0])[0] == "c"
    assert weighted_rrf([dense, bm25], [1.0, 0.0])[:3] == dense
    similarity = {"a": 0.80, "b": 0.795, "c": 0.70}
    assert dense_first(dense, similarity, bm25) == ["a", "b", "c", "d"]
    # b is within 0.01 of a, and BM25 ranks b higher, so b moves up; c is outside the band.
    assert dense_first(dense, similarity, bm25, tie_eps=0.01) == ["b", "a", "c", "d"]


class HashEmbedder:
    """Deterministic bag-of-words vectors: no model download in tests."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


def _row(record_id: str, start: str, end: str | None, text: str) -> dict:
    path = record_id.split("@")[0]
    return {
        "id": record_id,
        "record_id": record_id,
        "path": path,
        "act_id": "TEST",
        "doc_type": "act",
        "node_type": "clause",
        "effective_from": start,
        "effective_to": end or OPEN_END,
        "in_force_since": start,
        "text": text,
        "servable": "1",
    }


class _TableNamed:
    """Wraps a store so tests write to a throwaway table instead of the real index."""

    def __init__(self, store, table):
        self.store, self.table, self.backend = store, table, store.backend

    def write(self, table, rows, vectors):
        self.store.write(self.table, rows, vectors)

    def search(self, table, vector, k, **kwargs):
        return self.store.search(self.table, vector, k, **kwargs)

    def rows(self, table):
        return self.store.rows(self.table)


def _pg_store():
    try:
        return PgVectorStore(DEFAULT_PG_DSN)
    except Exception:
        pytest.skip("Postgres with pgvector not reachable (docker compose up -d)")


@pytest.fixture(params=["lancedb", "pgvector"])
def retriever(request, tmp_path):
    # Synthetic rows exercising the mechanics only (not legal text).
    rows = [
        _row("T/s1/ss1@v1", "2018-01-01", "2019-12-31", "alpha beta gamma old wording"),
        _row("T/s1/ss1@v2", "2020-01-01", None, "alpha beta gamma new wording"),
        _row("T/s2@v1", "2018-01-01", None, "delta epsilon unrelated text"),
        _row("CGST/s99/ss2@v1", "2018-01-01", None, "zeta eta theta"),
    ]
    if request.param == "lancedb":
        store = LanceStore(tmp_path / "lance")
    else:
        store = _TableNamed(_pg_store(), f"test_{tmp_path.name}".lower().replace("-", "_"))
    embedder = HashEmbedder()
    store.write(PROVISIONS_TABLE, rows, embedder.encode([row["text"] for row in rows]))
    yield Retriever(store, embedder)
    if request.param == "pgvector":
        store.store.conn.execute(f'DROP TABLE IF EXISTS "{store.table}"')


def test_date_filter_keeps_only_the_version_valid_on_the_reference_date(retriever) -> None:
    config = RetrievalConfig("t", dense=True, bm25=True, date_filter=True)
    early = retriever.search("alpha beta gamma wording", config, on=date(2019, 6, 1))
    late = retriever.search("alpha beta gamma wording", config, on=date(2021, 6, 1))
    assert "T/s1/ss1@v1" in early and "T/s1/ss1@v2" not in early
    assert "T/s1/ss1@v2" in late and "T/s1/ss1@v1" not in late


def test_without_date_filter_both_versions_compete(retriever) -> None:
    config = RetrievalConfig("t", dense=True, bm25=True)
    ranked = retriever.search("alpha beta gamma wording", config, on=date(2019, 6, 1))
    assert {"T/s1/ss1@v1", "T/s1/ss1@v2"} <= set(ranked)


def test_bm25_alone_and_dense_alone_both_rank(retriever) -> None:
    for config in (RetrievalConfig("b", dense=False, bm25=True), RetrievalConfig("d")):
        assert retriever.search("delta epsilon", config, on=date(2021, 1, 1))[0] == "T/s2@v1"


def test_exact_match_pins_the_cited_provision_first(retriever) -> None:
    query = "what does section 99(2) provide about alpha beta gamma"
    plain = retriever.search(query, RetrievalConfig("t", bm25=True), on=date(2021, 1, 1))
    pinned = retriever.search(
        query, RetrievalConfig("t", bm25=True, exact_match=True), on=date(2021, 1, 1)
    )
    assert plain[0] != "CGST/s99/ss2@v1"
    assert pinned[0] == "CGST/s99/ss2@v1"


def test_split_bm25_gives_rate_rows_their_own_index_and_ranking(tmp_path) -> None:
    # Synthetic rows exercising the mechanics only (not legal text): two long provision rows
    # and three short rows of one rate schedule, which mention "alpha" as often as T/s2 does.
    def rate(s_no: int, text: str) -> dict:
        row = _row("CTR/0-0000/schI@v1", "2018-01-01", None, text)
        return {**row, "id": f"{row['id']}#{s_no}", "act_id": "CTR", "doc_type": "notification"}

    filler = " ".join(f"word{n}" for n in range(40))
    rows = [
        _row("T/s1@v1", "2018-01-01", None, f"alpha beta beta {filler}"),
        _row("T/s2@v1", "2018-01-01", None, f"alpha gamma {filler}"),
        rate(1, "alpha one"),
        rate(2, "alpha two"),
        rate(3, "beta three"),
    ]
    embedder = HashEmbedder()
    store = LanceStore(tmp_path / "lance")
    store.write(PROVISIONS_TABLE, rows, embedder.encode([row["text"] for row in rows]))
    retriever = Retriever(store, embedder)
    merged = RetrievalConfig("b", dense=False, bm25=True)
    split = RetrievalConfig("b", dense=False, bm25=True, split_bm25=True)

    assert [len(part) for part, _ in retriever._bm25_indexes(PROVISIONS_TABLE, True)] == [2, 3]
    assert [len(part) for part, _ in retriever._bm25_indexes(PROVISIONS_TABLE, False)] == [5]
    # One index: the short rate rows outscore the long provision rows on the shared term.
    assert retriever.search("alpha", merged, on=date(2021, 1, 1))[0] == "CTR/0-0000/schI@v1"
    # Split: each index ranks its own rows, and both rankings reach the fused result.
    rankings = retriever._bm25_rankings(PROVISIONS_TABLE, "alpha beta", split, None)
    assert rankings == [["T/s1@v1", "T/s2@v1"], ["CTR/0-0000/schI@v1"]]
    fused = retriever.search("alpha beta", split, on=date(2021, 1, 1))
    # Equal RRF scores (rank 1 in each list) tie-break by id.
    assert fused == ["CTR/0-0000/schI@v1", "T/s1@v1", "T/s2@v1"]
    with_dense = RetrievalConfig("t", bm25=True, split_bm25=True)
    assert set(retriever.search("alpha beta", with_dense, on=date(2021, 1, 1))) == set(fused)
