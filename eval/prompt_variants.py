"""Generator prompt-variant experiment on the dev split (judge v2).

Each variant (generation.prompts.PROMPT_VARIANTS) is run on every dev item as a tagged run,
``pv-<variant>``, then judged with v2::

    python -m eval.generation --split dev --tag pv-<variant> --prompt-variant <variant> --no-judge
    python -m eval.prompt_variants --time --prescreen     # no judge: picks one challenger
    python -m eval.generation --split dev --tag pv-<variant> --from-items --rejudge --judge v2
    python -m eval.prompt_variants                        # baseline vs judged challenger

To keep judge cost down only the baseline and the single best challenger of the pre-screen
(see PRESCREEN_COST_RATIO) are judged.

This script reads the judged per-item files and reports, per variant: correct rate with its
95% bootstrap interval, citation precision and recall, abstention F1, partially_correct count,
mean generator input/output tokens and mean generator latency, over the items that reached the
generator. Output tokens include thinking tokens when the model reports them.

Latency comes from the provider call as recorded in the cache. Answers cached before latency
was recorded (the baseline reuses earlier answers) have none; ``--time`` re-sends those prompts
with the cache bypassed, only to time them (the answers are discarded), and stores the timings
in eval/results/prompt_variants_latency_dev.json.

Winner: highest correct rate; ties broken by citation recall, then fewer mean tokens. A variant
other than the baseline is adopted only if its correct rate is above the baseline's 95% interval,
or, when the gain is inside that interval, if it is also cheaper (fewer mean tokens).
"""

from __future__ import annotations

import csv
import json
from typing import Any

from eval.generation import ROOT, items_path, summarize
from generation.prompts import PROMPT_VARIANTS

SPLIT = "dev"
JUDGE = "v2"
OUT = ROOT / "eval" / "results" / f"prompt_variants_{SPLIT}.csv"
PRESCREEN_OUT = ROOT / "eval" / "results" / f"prompt_variants_prescreen_{SPLIT}.csv"
LATENCY_FILE = ROOT / "eval" / "results" / f"prompt_variants_latency_{SPLIT}.json"


def tag(variant: str) -> str:
    return f"pv-{variant}"


def load(variant: str, judged: bool = True) -> tuple[dict, list[dict[str, Any]]]:
    path = items_path(SPLIT, tag(variant), JUDGE if judged else "v1")
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


# --- Pre-screen without the judge ---------------------------------------------------------
# A challenger is dropped if its citation recall is below the baseline's, if it abstains on
# more answerable items (abstention FP), or if it costs more than PRESCREEN_COST_RATIO times
# the baseline's mean tokens without a citation-recall gain. Of the survivors only the best
# (citation recall, then citation precision, then fewer abstention FP, then fewer format
# failures, then fewer tokens) is judged.
PRESCREEN_COST_RATIO = 1.3
FORMAT_FAILURES = ("invalid_output", "no_supported_claims")


def prescreen_row(variant: str, timings: dict[str, float]) -> dict[str, Any]:
    _, records = load(variant, judged=False)
    overall = next(
        row
        for row in summarize(records, {})
        if (row["source"], row["category"]) == ("all", "overall")
    )
    generated = [r for r in records if r.get("llm")]
    usages = [u for u in map(_usage, generated) if u]
    latencies = [
        r["llm"]["latency_s"] if r["llm"].get("latency_s") is not None else timings.get(r["key"])
        for r in generated
    ]
    latencies = [value for value in latencies if value is not None]
    mean_in = _mean([u[0] for u in usages])
    mean_out = _mean([u[1] for u in usages])
    return {
        "variant": variant,
        "generated": len(generated),
        "citation_precision": overall["citation_precision"],
        "citation_recall": overall["citation_recall"],
        "abstain_tp": overall["abstain_tp"],
        "abstain_fp": overall["abstain_fp"],
        "abstain_fn": overall["abstain_fn"],
        "format_failures": sum(1 for r in records if r["abstain_reason"] in FORMAT_FAILURES),
        "dropped_citations": sum(len(r.get("dropped_citations") or []) for r in records),
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "mean_tokens": round(mean_in + mean_out, 1) if usages else "n/a",
        "mean_latency_s": _mean(latencies),
        "latency_n": len(latencies),
    }


def prescreen(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Rows with a ``prescreen`` verdict, and the single challenger to judge (or None)."""
    base = next(row for row in rows if row["variant"] == "baseline")
    survivors = []
    for row in rows:
        if row["variant"] == "baseline":
            row["prescreen"] = "baseline"
            continue
        reasons = []
        if row["citation_recall"] < base["citation_recall"]:
            reasons.append("citation recall below baseline")
        if row["abstain_fp"] > base["abstain_fp"]:
            reasons.append("more abstention FP than baseline")
        dearer = row["mean_tokens"] > PRESCREEN_COST_RATIO * base["mean_tokens"]
        if dearer and row["citation_recall"] <= base["citation_recall"]:
            reasons.append(f">{PRESCREEN_COST_RATIO - 1:.0%} more tokens without a citation gain")
        row["prescreen"] = "dropped: " + "; ".join(reasons) if reasons else "survives"
        if not reasons:
            survivors.append(row)
    if not survivors:
        return rows, None
    best = min(
        survivors,
        key=lambda r: (
            -r["citation_recall"],
            -r["citation_precision"],
            r["abstain_fp"],
            r["format_failures"],
            r["mean_tokens"],
        ),
    )
    best["prescreen"] = "survives: judged (best challenger)"
    return rows, best["variant"]


def _usage(record: dict[str, Any]) -> tuple[int, int] | None:
    usage = (record.get("llm") or {}).get("usage") or {}
    if "promptTokenCount" not in usage:
        return None
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return usage["promptTokenCount"], output


def _mean(values: list[float]) -> float | str:
    return round(sum(values) / len(values), 3) if values else "n/a"


def variant_row(variant: str, timings: dict[str, float]) -> dict[str, Any]:
    _, records = load(variant)
    overall = next(
        row
        for row in summarize(records, {})
        if (row["source"], row["category"]) == ("all", "overall")
    )
    generated = [r for r in records if r.get("llm")]
    usages = [u for u in map(_usage, generated) if u]
    latencies = []
    for record in generated:
        latency = record["llm"].get("latency_s")
        latency = latency if latency is not None else timings.get(record["key"])
        if latency is not None:
            latencies.append(latency)
    mean_in = _mean([u[0] for u in usages])
    mean_out = _mean([u[1] for u in usages])
    return {
        "variant": variant,
        "n": overall["n"],
        "generated": len(generated),
        "correct": overall["correct"],
        "correct_rate": overall["correct_rate"],
        "correct_rate_ci_low": overall["correct_rate_ci_low"],
        "correct_rate_ci_high": overall["correct_rate_ci_high"],
        "citation_precision": overall["citation_precision"],
        "citation_recall": overall["citation_recall"],
        "abstain_f1": overall["abstain_f1"],
        "judge_correct": overall["judge_correct"],
        "partially_correct": overall["judge_partially_correct"],
        "judge_incorrect": overall["judge_incorrect"],
        "judge_error": overall["judge_error"],
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "mean_tokens": round(mean_in + mean_out, 1) if usages else "n/a",
        "mean_latency_s": _mean(latencies),
        "latency_n": len(latencies),
    }


def choose(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(best variant by the ranking, adopted variant, reason)."""

    def rank(row: dict[str, Any]) -> tuple:
        recall = row["citation_recall"] if isinstance(row["citation_recall"], float) else 0.0
        tokens = row["mean_tokens"] if isinstance(row["mean_tokens"], float) else float("inf")
        return (-row["correct_rate"], -recall, tokens)

    best = min(rows, key=rank)
    base = next(row for row in rows if row["variant"] == "baseline")
    if best["variant"] == "baseline":
        return "baseline", "baseline", "the baseline ranks first"
    gain = best["correct_rate"] - base["correct_rate"]
    if best["correct_rate"] > base["correct_rate_ci_high"]:
        reason = (
            f"{best['variant']} correct rate {best['correct_rate']} is above the baseline's "
            f"95% interval [{base['correct_rate_ci_low']}, {base['correct_rate_ci_high']}]"
        )
        return best["variant"], best["variant"], reason
    cheaper = rank(best)[2] < rank(base)[2]
    inside = (
        f"its gain of {gain:+.3f} is inside the baseline's 95% interval "
        f"[{base['correct_rate_ci_low']}, {base['correct_rate_ci_high']}]"
    )
    tokens = f"{best['mean_tokens']} vs {base['mean_tokens']} mean tokens"
    if cheaper:
        reason = f"{best['variant']} ranks first; {inside}, but it is cheaper ({tokens})"
        return best["variant"], best["variant"], reason
    reason = (
        f"{best['variant']} ranks first, but {inside} and it is not cheaper ({tokens}): "
        "keep the baseline"
    )
    return best["variant"], "baseline", reason


def time_uncached(variant: str) -> dict[str, float]:
    """Re-send a variant's generator prompts with the cache bypassed and time them."""
    from datetime import date

    from generation.context import Chunk
    from generation.llm import generator_client
    from generation.prompts import generator_prompt, generator_system
    from generation.schemas import response_schema
    from retrieval.pipeline import PROVISIONS_TABLE
    from retrieval.store import open_store

    _, records = load(variant, judged=False)
    rows = {row["id"]: row for row in open_store().rows(PROVISIONS_TABLE)}
    client = generator_client(cache_dir=None)
    system, schema = generator_system(variant), response_schema(variant)
    timings: dict[str, float] = {}
    for record in records:
        if not record.get("llm"):
            continue
        chunks = [
            Chunk(
                cid,
                rows[cid]["record_id"],
                rows[cid]["path"],
                rows[cid]["doc_type"],
                rows[cid]["effective_from"],
                rows[cid]["effective_to"],
                rows[cid]["text"],
                0.0,
                False,
            )
            for cid in record["context_ids"]
        ]
        prompt = generator_prompt(
            record["question"], date.fromisoformat(record["ref_date"]), chunks
        )
        response = client.complete(system, prompt, schema=schema)
        timings[record["key"]] = response.latency_s
    return timings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare generator prompt variants (dev).")
    parser.add_argument(
        "--time", action="store_true", help="time variants whose cached answers lack latency"
    )
    parser.add_argument(
        "--prescreen", action="store_true", help="compare unjudged variants with the baseline"
    )
    args = parser.parse_args()
    timings: dict[str, dict[str, float]] = {}
    if LATENCY_FILE.exists():
        timings = json.loads(LATENCY_FILE.read_text(encoding="utf-8"))
    if args.time:
        for variant in PROMPT_VARIANTS:
            _, records = load(variant, judged=False)
            if any(r.get("llm") and r["llm"].get("latency_s") is None for r in records):
                timings[variant] = time_uncached(variant)
                print(f"timed {len(timings[variant])} uncached calls for {variant}")
        LATENCY_FILE.write_text(json.dumps(timings, indent=1), encoding="utf-8")
    if args.prescreen:
        rows = [prescreen_row(variant, timings.get(variant, {})) for variant in PROMPT_VARIANTS]
        rows, challenger = prescreen(rows)
        _write(PRESCREEN_OUT, rows)
        for row in rows:
            print(
                f"{row['variant']:17} citP {row['citation_precision']} "
                f"citR {row['citation_recall']} abst TP/FP/FN {row['abstain_tp']}/"
                f"{row['abstain_fp']}/{row['abstain_fn']} format {row['format_failures']} "
                f"tokens {row['mean_input_tokens']}/{row['mean_output_tokens']} "
                f"latency {row['mean_latency_s']}s (n={row['latency_n']}) -> {row['prescreen']}"
            )
        print(f"challenger to judge: {challenger}; wrote {PRESCREEN_OUT.relative_to(ROOT)}")
        return
    # Final comparison: the baseline and every variant judged with v2.
    judged = [v for v in PROMPT_VARIANTS if items_path(SPLIT, tag(v), JUDGE).exists()]
    rows = [variant_row(variant, timings.get(variant, {})) for variant in judged]
    _write(OUT, rows)
    for row in rows:
        print(
            f"{row['variant']:17} correct {row['correct']}/{row['n']} ({row['correct_rate']}) "
            f"CI [{row['correct_rate_ci_low']}, {row['correct_rate_ci_high']}] "
            f"citP {row['citation_precision']} citR {row['citation_recall']} "
            f"abstF1 {row['abstain_f1']} partial {row['partially_correct']} "
            f"tokens {row['mean_input_tokens']}/{row['mean_output_tokens']} "
            f"latency {row['mean_latency_s']}s (n={row['latency_n']})"
        )
    best, adopted, reason = choose(rows)
    print(f"best: {best}; adopt: {adopted}. {reason}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def _write(path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
