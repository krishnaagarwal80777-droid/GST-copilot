"""Observability settings: request-log path and the per-model price table."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = Path(os.environ.get("GST_REQUEST_LOG", ROOT / "logs" / "requests.jsonl"))

# USD per 1M tokens (input, output), standard paid tier. Taken from public price listings in
# November 2026 (aggregators quoting ai.google.dev/gemini-api/docs/pricing); check that page
# before relying on the numbers. Gemini bills thinking tokens as output. The free tier this
# project runs on costs nothing, so this is the cost at list price. A model not listed here
# gets cost None.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    if model is None:
        return 0.0
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 8)
