"""One JSON line per /ask request.

Fields: request_id, timestamp (UTC), question_sha256 and question_chars (never the text),
reference_date, knowledge_date, timings_ms (retrieve, rerank, generate, verify, total; a stage
that did not run is absent), retrieved_ids (the generator context), gate_score (best
cross-encoder score), abstained, abstain_reason, guardrail, model, input_tokens,
output_tokens, cost_usd, cache_hit, status ("ok" or "error"). With calculator routing:
tool_calls (one {name, latency_ms, success} per call), planner_calls, planner_input_tokens,
planner_output_tokens and planner_cost_usd; cost_usd then includes the planner turns. No API
keys, prompts or model output are written (tool arguments and results are not logged).

cost_usd is the list-price cost of the generator call (observability.config); 0 when no call
was made or the answer came from the disk cache, None for a model without a price. Tokens are
those the provider reported, also for a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from observability.config import LOG_PATH, estimate_cost


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def token_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    """(input, output) from Gemini usageMetadata or OpenAI-style usage."""
    usage = usage or {}
    if "promptTokenCount" in usage or "candidatesTokenCount" in usage:
        output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return int(usage.get("promptTokenCount", 0)), int(output)
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def llm_fields(llm: dict[str, Any] | None) -> dict[str, Any]:
    if not llm:
        return {
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cache_hit": None,
        }
    input_tokens, output_tokens = token_counts(llm.get("usage"))
    cached = bool(llm.get("cached"))
    cost = 0.0 if cached else estimate_cost(llm.get("model"), input_tokens, output_tokens)
    return {
        "model": llm.get("model"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "cache_hit": cached,
    }


def planner_fields(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Token and cost totals of the calculator planner's turns (a cached turn costs 0)."""
    tokens_in = tokens_out = 0
    cost: float | None = 0.0
    for turn in turns:
        fields = llm_fields(turn)
        tokens_in += fields["input_tokens"]
        tokens_out += fields["output_tokens"]
        cost = None if cost is None or fields["cost_usd"] is None else cost + fields["cost_usd"]
    return {
        "planner_calls": len(turns),
        "planner_input_tokens": tokens_in,
        "planner_output_tokens": tokens_out,
        "planner_cost_usd": None if cost is None else round(cost, 8),
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RequestLogger:
    def __init__(self, path: Path = LOG_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
