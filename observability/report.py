"""Summary of logs/requests.jsonl: latency per stage, abstention and guardrail rates, tokens and
cost per request.

    python -m observability.report [--log PATH]

Latency percentiles are over the requests in which the stage ran. The abstention rate is over
requests that passed the input guardrails; the guardrail rate is over all requests. Tokens
are averaged over requests that reached the generator; cost over all requests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generation.timing import STAGES
from observability.config import LOG_PATH


def load(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a crash
    return records


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    latency = {}
    for stage in (*STAGES, "total"):
        values = [r["timings_ms"][stage] for r in ok if stage in r.get("timings_ms", {})]
        latency[stage] = {
            "n": len(values),
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
        }
    guarded = [r for r in ok if r.get("guardrail")]
    passed = [r for r in ok if not r.get("guardrail")]
    abstained = [r for r in passed if r.get("abstained")]
    generated = [r for r in ok if r.get("model")]
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    return {
        "requests": len(records),
        "errors": len(records) - len(ok),
        "latency": latency,
        "guardrail_rate": _rate(len(guarded), len(ok)),
        "guardrails": dict(Counter(r["guardrail"] for r in guarded)),
        "abstention_rate": _rate(len(abstained), len(passed)),
        "abstain_reasons": dict(Counter(r.get("abstain_reason") for r in abstained)),
        "generator_calls": len(generated),
        "cache_hit_rate": _rate(sum(bool(r.get("cache_hit")) for r in generated), len(generated)),
        "mean_input_tokens": (
            round(sum(r["input_tokens"] for r in generated) / len(generated), 1)
            if generated
            else None
        ),
        "mean_output_tokens": (
            round(sum(r["output_tokens"] for r in generated) / len(generated), 1)
            if generated
            else None
        ),
        "mean_cost_usd": round(sum(costs) / len(costs), 8) if costs else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "unpriced_requests": sum(r.get("cost_usd") is None for r in ok),
    }


def format_report(summary: dict[str, Any]) -> str:
    def show(value: Any) -> str:
        return "-" if value is None else str(value)

    lines = [f"requests: {summary['requests']} (errors: {summary['errors']})", ""]
    lines.append(f"{'stage':10} {'n':>6} {'p50 ms':>10} {'p95 ms':>10}")
    for stage, row in summary["latency"].items():
        lines.append(
            f"{stage:10} {row['n']:>6} {show(row['p50_ms']):>10} {show(row['p95_ms']):>10}"
        )
    lines += [
        "",
        f"guardrail rate:   {show(summary['guardrail_rate'])} {summary['guardrails']}",
        f"abstention rate:  {show(summary['abstention_rate'])} {summary['abstain_reasons']}",
        f"generator calls:  {summary['generator_calls']} "
        f"(cache hit rate {show(summary['cache_hit_rate'])})",
        f"tokens/request:   input {show(summary['mean_input_tokens'])}, "
        f"output {show(summary['mean_output_tokens'])} (requests that reached the generator)",
        f"cost/request:     ${show(summary['mean_cost_usd'])} "
        f"(total ${show(summary['total_cost_usd'])}; unpriced {summary['unpriced_requests']})",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    args = parser.parse_args()
    records = load(args.log)
    if not records:
        print(f"no requests logged in {args.log}")
        return
    print(format_report(summarize(records)))


if __name__ == "__main__":
    main()
