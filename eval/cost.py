"""Token and cost log for hosted LLM calls, read from the disk cache (no API calls).

Every uncached call writes one cache file (generation.llm) with the provider's usage. A step
is the set of cache files written between two timestamps; ``python -m eval.cost --step NAME
--since ISO`` appends its calls, tokens and estimated cost to eval/results/cost_log.csv and
prints the running total.

Prices are per million tokens, as listed by the providers (assumed, not read from a bill):
Groq openai/gpt-oss-120b $0.15 input / $0.75 output. Gemini calls run on the free tier and
are logged at $0.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from generation.llm import CACHE_DIR

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "eval" / "results" / "cost_log.csv"
PRICES = {"openai/gpt-oss-120b": (0.15, 0.75), "openai/gpt-oss-20b": (0.075, 0.30)}
BUDGET_STOP = 2.00  # dollars: stop the experiment once the logged total passes this
FIELDS = (
    "step",
    "since",
    "until",
    "groq_calls",
    "groq_input_tokens",
    "groq_output_tokens",
    "groq_cost_usd",
    "gemini_calls",
    "gemini_input_tokens",
    "gemini_output_tokens",
)


def usage_since(since: datetime, until: datetime | None = None) -> dict:
    until = until or datetime.now(UTC)
    out = dict.fromkeys(FIELDS[3:], 0)
    out["groq_cost_usd"] = 0.0
    for path in CACHE_DIR.glob("*.json"):
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if not since <= stamp <= until:
            continue
        response = json.loads(path.read_text(encoding="utf-8"))["response"]
        usage = response.get("usage") or {}
        if response["provider"] == "groq":
            tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            price_in, price_out = PRICES.get(response["model"], PRICES["openai/gpt-oss-120b"])
            out["groq_calls"] += 1
            out["groq_input_tokens"] += tokens_in
            out["groq_output_tokens"] += tokens_out
            out["groq_cost_usd"] += (tokens_in * price_in + tokens_out * price_out) / 1e6
        else:
            out["gemini_calls"] += 1
            out["gemini_input_tokens"] += usage.get("promptTokenCount", 0)
            out["gemini_output_tokens"] += usage.get("candidatesTokenCount", 0) + usage.get(
                "thoughtsTokenCount", 0
            )
    out["groq_cost_usd"] = round(out["groq_cost_usd"], 4)
    return out


def total_cost(path: Path = LOG) -> float:
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8", newline="") as handle:
        return round(sum(float(row["groq_cost_usd"]) for row in csv.DictReader(handle)), 4)


def log_step(step: str, since: datetime, until: datetime | None = None) -> dict:
    until = until or datetime.now(UTC)
    row = {
        "step": step,
        "since": since.isoformat(timespec="seconds"),
        "until": until.isoformat(timespec="seconds"),
        **usage_since(since, until),
    }
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Log tokens and cost of one step.")
    parser.add_argument("--step", required=True)
    parser.add_argument("--since", required=True, help="ISO timestamp (UTC) the step started")
    parser.add_argument("--until", help="ISO timestamp (UTC) the step ended (default: now)")
    args = parser.parse_args()
    since = datetime.fromisoformat(args.since).astimezone(UTC)
    until = datetime.fromisoformat(args.until).astimezone(UTC) if args.until else None
    row = log_step(args.step, since, until)
    total = total_cost()
    print(
        f"{row['step']}: groq {row['groq_calls']} calls, {row['groq_input_tokens']} in / "
        f"{row['groq_output_tokens']} out, ${row['groq_cost_usd']}; gemini "
        f"{row['gemini_calls']} calls. Logged total ${total}"
        + (f" -- over the ${BUDGET_STOP} stop" if total > BUDGET_STOP else "")
    )


if __name__ == "__main__":
    main()
