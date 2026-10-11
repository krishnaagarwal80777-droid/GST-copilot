"""Vector store behind one interface: pgvector when Postgres is up, else embedded LanceDB.

Rows carry id, record_id, path, act_id, doc_type, node_type, the bitemporal fields and text.
Dates are ISO strings; an open-ended effective_to is stored as OPEN_END so the date filter is
one plain comparison in either backend:

    effective_from <= on AND on <= effective_to AND in_force_since <= known_by
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from retrieval.corpus import IndexRow

ROOT = Path(__file__).resolve().parents[1]
LANCE_DIR = ROOT / "data" / "index" / "lancedb"
OPEN_END = "9999-12-31"
DEFAULT_PG_DSN = "postgresql://gst_copilot:local_dev_only@localhost:5432/gst_copilot"
COLUMNS = (
    "id",
    "record_id",
    "path",
    "act_id",
    "doc_type",
    "node_type",
    "effective_from",
    "effective_to",
    "in_force_since",
    "text",
    "servable",
)


def row_dict(row: IndexRow) -> dict[str, Any]:
    data = asdict(row)
    for key in ("effective_from", "in_force_since"):
        data[key] = data[key].isoformat()
    data["effective_to"] = row.effective_to.isoformat() if row.effective_to else OPEN_END
    data["servable"] = "1" if row.servable else "0"
    return data


class VectorStore(Protocol):
    backend: str

    def write(self, table: str, rows: Sequence[dict[str, Any]], vectors: np.ndarray) -> None: ...

    def search(
        self,
        table: str,
        vector: np.ndarray,
        k: int,
        *,
        on: date | None = None,
        known_by: date | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k rows by cosine similarity; with ``on``, only rows valid on that date."""
        ...

    def rows(self, table: str) -> list[dict[str, Any]]: ...


def _date_clause(on: date | None, known_by: date | None) -> str | None:
    if on is None:
        return None
    day = on.isoformat()
    known = (known_by or date.today()).isoformat()
    return f"effective_from <= '{day}' AND effective_to >= '{day}' AND in_force_since <= '{known}'"


class LanceStore:
    backend = "lancedb"

    def __init__(self, directory: Path = LANCE_DIR) -> None:
        import lancedb

        directory.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(directory))

    def write(self, table: str, rows: Sequence[dict[str, Any]], vectors: np.ndarray) -> None:
        import pyarrow as pa

        vectors = np.asarray(vectors, dtype=np.float32)
        columns = {
            column: pa.array([row[column] for row in rows], pa.string()) for column in COLUMNS
        }
        columns["vector"] = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), vectors.shape[1]
        )
        self.db.create_table(table, data=pa.table(columns), mode="overwrite")

    def search(self, table, vector, k, *, on=None, known_by=None):
        query = self.db.open_table(table).search(vector.tolist()).metric("cosine")
        clause = _date_clause(on, known_by)
        if clause:
            query = query.where(clause, prefilter=True)
        hits = query.limit(k).to_list()
        return [{key: hit[key] for key in hit if key != "vector"} for hit in hits]

    def rows(self, table: str) -> list[dict[str, Any]]:
        frame = self.db.open_table(table).to_arrow().drop(["vector"])
        return frame.to_pylist()


class PgVectorStore:
    """pgvector backend (compose.yaml). Not exercised where Postgres is unavailable."""

    backend = "pgvector"

    def __init__(self, dsn: str) -> None:
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    @staticmethod
    def _literal(vector: np.ndarray) -> str:
        return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"

    def write(self, table, rows, vectors):
        columns = ", ".join(f"{column} text" for column in COLUMNS)
        with self.conn.cursor() as cursor:
            cursor.execute(f'DROP TABLE IF EXISTS "{table}"')
            dimension = int(np.asarray(vectors).shape[1])
            cursor.execute(f'CREATE TABLE "{table}" ({columns}, embedding vector({dimension}))')
            placeholders = ", ".join(["%s"] * (len(COLUMNS) + 1))
            cursor.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [
                    # Postgres text cannot hold NUL bytes (pypdf emits some in raw PDF text).
                    (
                        *[str(row[column]).replace("\x00", "") for column in COLUMNS],
                        self._literal(vector),
                    )
                    for row, vector in zip(rows, vectors, strict=True)
                ],
            )

    def search(self, table, vector, k, *, on=None, known_by=None):
        clause = _date_clause(on, known_by)
        where = f"WHERE {clause}" if clause else ""
        sql = (
            f'SELECT {", ".join(COLUMNS)}, embedding <=> %s::vector AS _distance FROM "{table}" '
            f"{where} ORDER BY _distance LIMIT %s"
        )
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (self._literal(vector), k))
            names = [description.name for description in cursor.description]
            return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]

    def rows(self, table):
        with self.conn.cursor() as cursor:
            cursor.execute(f'SELECT {", ".join(COLUMNS)} FROM "{table}"')
            return [dict(zip(COLUMNS, values, strict=True)) for values in cursor.fetchall()]


def open_store() -> VectorStore:
    """pgvector if Postgres answers (``docker compose up -d``), otherwise embedded LanceDB."""
    try:
        return PgVectorStore(os.environ.get("GST_PG_DSN", DEFAULT_PG_DSN))
    except Exception:
        return LanceStore()
