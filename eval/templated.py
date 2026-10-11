"""Templated eval items generated from verified data (eval/questions_templated.jsonl).

Two templates, each producing a linked pair (same question, two reference dates):

- version_change: one pair per verified version change of an indexed Act/Rule path, asked on
  the last day of the old version and the first day of the new one. Gold answer = the text of
  the version valid on that date; gold path = the provision path.
- rate_change: one pair per goods-schedule row whose central tax rate differs between
  Notification 1/2017-CT(Rate) as issued (asked on 2017-07-01) and 9/2025-CT(Rate) (asked on
  2025-10-01). A row pair is matched only when its tariff entry is identical and appears once in
  each notification, and its description is word-for-word identical. Gold answer = the rate and
  the quoted description; gold path = that notification's schedule record.

Exclusions, so templated and hand-written metrics do not overlap:
- version boundaries already asked by a hand-written date pair, and rate rows asked by a
  hand-written rate pair;
- Act versions that begin after the Act source horizon (eval.harness.act_source_horizon: the
  date of the latest CBIC section-page snapshot), as for the hand-written pending_source items;
- versions whose text is identical to their neighbour's.

The 227 changed rate rows that match (descriptions of 6+ words) are listed by ``--list-rates``;
the pairs written are a deterministic sample of RATE_PAIRS rows, stratified by (old rate, new
rate), so the templated set stays near 100 items.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from eval.harness import DEFAULT_QUESTION_FILE, act_source_horizon, load_eval_items
from gst_copilot.models import EvalItem, ProvisionRecord
from ingest.index import load_processed_records
from ingest.rate_schedule import RateRow, all_rows
from retrieval.corpus import CORE_FILE, ENGINE_FILE, citation_label

ROOT = Path(__file__).resolve().parents[1]
TEMPLATED_FILE = ROOT / "eval" / "questions_templated.jsonl"
RATE_DATES = (date(2017, 7, 1), date(2025, 10, 1))
RATE_PAIRS = 15
SEED = 20261011
# Act versions starting after this date are left out (kept as a name for callers).
ACT_SOURCE_HORIZON = act_source_horizon()
CHANGED_TEXT_CAP = 1500
_WORDS = re.compile(r"[a-z0-9]+")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def indexed_versions() -> dict[str, list[ProvisionRecord]]:
    """Verified, non-empty Act/Rule versions as retrieval.corpus indexes them, per path."""
    core = [r for r in load_processed_records(CORE_FILE) if r.verified and r.text.strip()]
    core_paths = {r.path for r in core}
    engine = [
        r
        for r in load_processed_records(ENGINE_FILE)
        if r.verified and r.doc_type == "rule" and r.path not in core_paths and r.text.strip()
    ]
    chains: dict[str, list[ProvisionRecord]] = defaultdict(list)
    for record in core + engine:
        chains[record.path].append(record)
    return {path: sorted(v, key=lambda r: r.effective_from) for path, v in chains.items()}


def changed_text(this: str, other: str) -> str:
    """Words of ``this`` that ``other`` lacks, as ' … '-joined spans with a little context."""
    import difflib

    a, b = this.split(), other.split()
    spans = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            spans.append(" ".join(a[max(i1 - 6, 0) : i2 + 6]))
    joined = " … ".join(spans)
    return joined[:CHANGED_TEXT_CAP] + (" …" if len(joined) > CHANGED_TEXT_CAP else "")


def question_for(record: ProvisionRecord) -> str:
    label = citation_label(record.path)
    source = "the CGST Act" if record.doc_type == "act" else "the CGST Rules"
    if ", Explanation " in label:
        base, number = label.split(", Explanation ")
        return f"What does Explanation {number} to {base} of {source} provide?"
    if ", proviso " in label:
        base, number = label.split(", proviso ")
        return f"What does proviso {number} to {base} of {source} provide?"
    return f"What does {label} of {source} provide?"


def hand_boundaries(hand: list[EvalItem]) -> set[tuple[str, date]]:
    """(path, first day of the later version) for every hand-written date pair."""
    groups: dict[str, list[EvalItem]] = defaultdict(list)
    for item in hand:
        if item.category == "date_sensitive":
            groups[item.group_id].append(item)
    out = set()
    for members in groups.values():
        later = max(members, key=lambda item: item.ref_date)
        for path in {p for item in members for p in item.gold_paths}:
            out.add((path, later.ref_date))
    return out


def version_change_items(hand: list[EvalItem]) -> list[EvalItem]:
    covered = hand_boundaries(hand)
    horizon = ACT_SOURCE_HORIZON
    items: list[EvalItem] = []
    for path, chain in sorted(indexed_versions().items()):
        for old, new in zip(chain, chain[1:], strict=False):
            if (
                old.effective_to is None
                or old.effective_to + timedelta(days=1) != new.effective_from
            ):
                continue
            if (path, new.effective_from) in covered:
                continue
            if new.doc_type == "act" and new.effective_from > horizon:
                continue
            if _words(old.text) == _words(new.text):
                continue
            question = question_for(new)
            group = f"tpl-ds-{path.replace('/', '-')}-{new.effective_from.isoformat()}"
            for this, other, on in ((old, new, old.effective_to), (new, old, new.effective_from)):
                items.append(
                    EvalItem(
                        question=question,
                        ref_date=on,
                        gold_paths=[path],
                        gold_answer=_flat(this.text),
                        category="date_sensitive",
                        answerable=True,
                        source="templated",
                        group_id=group,
                        template={
                            "kind": "version_change",
                            "record_id": this.id,
                            "other_record_id": other.id,
                            "boundary": new.effective_from.isoformat(),
                            "amended_by": new.amended_by,
                            "verified_by": this.verified_by or "engine",
                            # What distinguishes this date's text from the other date's.
                            "only_in_this": changed_text(_flat(this.text), _flat(other.text)),
                            "only_in_other": changed_text(_flat(other.text), _flat(this.text)),
                        },
                    )
                )
    return items


def changed_rate_rows() -> list[tuple[RateRow, RateRow]]:
    """(1/2017 row, 9/2025 row) pairs with one shared tariff entry, same words, other rate."""
    rows = all_rows()
    old, new = rows["1/2017"], rows["9/2025"]

    def unique(side: list[RateRow]) -> dict[str, RateRow]:
        by_key: dict[str, list[RateRow]] = defaultdict(list)
        for row in side:
            if row.tariff:
                by_key[row.tariff_key].append(row)
        return {key: group[0] for key, group in by_key.items() if len(group) == 1}

    old_by, new_by = unique(old), unique(new)
    pairs = []
    for key in sorted(old_by.keys() & new_by.keys()):
        before, after = old_by[key], new_by[key]
        same_words = _words(before.description) == _words(after.description)
        if same_words and before.rate != after.rate and len(_words(after.description)) >= 6:
            pairs.append((before, after))
    return pairs


def _hand_rate_quotes(hand: list[EvalItem]) -> list[list[str]]:
    quotes = []
    for item in hand:
        if any(path.startswith("CTR/") for path in item.gold_paths):
            quotes.extend(_words(q) for q in re.findall(r'"([^"]+)"', item.gold_answer))
    return quotes


def sample_rate_pairs(
    pairs: list[tuple[RateRow, RateRow]], hand: list[EvalItem], n: int = RATE_PAIRS
) -> list[tuple[RateRow, RateRow]]:
    """Deterministic sample, stratified by (old rate, new rate), at least one per stratum."""
    quotes = _hand_rate_quotes(hand)
    pool = [p for p in pairs if _words(p[1].description) not in quotes]
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for pair in pool:
        strata[(pair[0].rate, pair[1].rate)].append(pair)
    total = len(pool)
    keys = sorted(strata)
    quota = {key: 1 for key in keys}
    remaining = n - len(keys)
    shares = {key: (len(strata[key]) / total) * remaining for key in keys}
    for key in keys:
        quota[key] += int(shares[key])
    left = n - sum(quota.values())
    for key in sorted(keys, key=lambda k: -(shares[k] - int(shares[k])))[:left]:
        quota[key] += 1
    rng = random.Random(SEED)
    chosen = []
    for key in keys:
        group = sorted(strata[key], key=lambda p: p[1].tariff_key)
        chosen.extend(rng.sample(group, min(quota[key], len(group))))
    return sorted(chosen, key=lambda p: p[1].tariff_key)


def _rate_answer(row: RateRow) -> str:
    return (
        f'{row.rate} per cent. central tax: "{row.description}" ({row.tariff}) is in Schedule '
        f"{row.schedule} of Notification No. {row.notification}-Central Tax (Rate), "
        f"S. No. {row.s_no}."
    )


def rate_change_items(pairs: list[tuple[RateRow, RateRow]]) -> list[EvalItem]:
    items = []
    for before, after in pairs:
        question = (
            "What is the rate of central tax on intra-State supply of goods described as "
            f'"{after.description}" ({after.tariff})?'
        )
        sides = {
            side: {
                "notification": row.notification,
                "schedule": row.schedule,
                "s_no": row.s_no,
                "rate": row.rate,
                "path": row.path,
            }
            for side, row in (("before", before), ("after", after))
        }
        for row, on in ((before, RATE_DATES[0]), (after, RATE_DATES[1])):
            items.append(
                EvalItem(
                    question=question,
                    ref_date=on,
                    gold_paths=[row.path],
                    gold_answer=_rate_answer(row),
                    category="numeric",
                    answerable=True,
                    expected_tool_output={"central_tax_rate": row.rate, "path": row.path},
                    source="templated",
                    group_id=f"tpl-rate-{after.tariff_key}",
                    template={"kind": "rate_change", "tariff": after.tariff, **sides},
                )
            )
    return items


def build() -> list[EvalItem]:
    hand = load_eval_items(DEFAULT_QUESTION_FILE)
    pairs = sample_rate_pairs(changed_rate_rows(), hand)
    return version_change_items(hand) + rate_change_items(pairs)


def write(items: list[EvalItem], path: Path = TEMPLATED_FILE) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            payload = item.model_dump(mode="json", exclude_none=True)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Generate templated eval items.")
    parser.add_argument("--list-rates", action="store_true", help="print all changed rate rows")
    args = parser.parse_args()
    if args.list_rates:
        for before, after in changed_rate_rows():
            print(
                f"{after.tariff:30} {before.rate:>5} -> {after.rate:<5} "
                f"{before.path} #{before.s_no} -> {after.path} #{after.s_no}"
            )
        return
    items = build()
    write(items)
    counts = Counter(
        (item.template["kind"], item.template.get("verified_by", "")) for item in items
    )
    print(f"wrote {len(items)} items to {TEMPLATED_FILE.relative_to(ROOT)}: {dict(counts)}")


if __name__ == "__main__":
    main()
