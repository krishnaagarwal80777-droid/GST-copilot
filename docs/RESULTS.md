# GST Tax Copilot: results

_Commit hashes in older result rows map to current ones via [eval/results/commit_map.csv](../eval/results/commit_map.csv)._

Final frozen evaluation of the citation-grounded GST assistant. Code was frozen at commit
`427dd71`, and the final test results were committed in `718b8fd`. Every number below comes from
a committed file in `eval/results/`. The file behind each table is named in an HTML comment
beside it. Where a figure comes from elsewhere (code or `data/`), the comment says so.

**Summary**

- On the 64 test items, overall correctness did not change significantly between the first
  pipeline and the final one: 53/64 in v1, 53/64 in final, with overlapping 95% intervals.
- What did improve on test: rate questions went from 13/16 to 16/16, false abstentions from
  5 to 1, and citation recall from 0.89 to 0.95.
- Retrieval: the date filter is the step that matters most. It lifts test MRR from 0.64 to
  0.98, which puts the version in force on the reference date at rank 1.
- The judge was validated against human labels on 34 dev rows. Judge v2 agrees on 32/34
  (Cohen's kappa 0.89); v1 agrees on 29/34 (kappa 0.72).
- Prompt optimisation found no significant gain, so the baseline prompt was kept.

---

## 1. Setup

### Corpus

| Source | What is indexed | Knowledge horizon |
|---|---|---|
| CGST Act, 2017 | Provision versions from July 2017, structured by section > sub-section > clause > proviso > explanation | 2026-10-12 |
| CGST Rules, 2017 | Rule versions built from the base text and the amendment notifications | 2026-05-07 |
| Central tax (rate) notifications | Goods rate schedules 1/2017 and 9/2025 and goods exemptions 2/2017 and 10/2025, one index row per schedule entry, as issued | 2026-01-31 |

<!-- Sources: horizons from generation/horizon.py (knowledge_horizons(): Act =
eval.harness.act_source_horizon(), Rules = latest issue_date in data/processed/amendments.jsonl,
rates = day before 19/2025 comes into force); indexed notifications from
retrieval/corpus.py RATE_NOTIFICATIONS; index size 3,777 rows from eval/results/retrieval.csv
(index_size). These are code and data, not eval/results files. -->

- The structured index has 3,777 rows. Every row carries `effective_from`, `effective_to` and
  `in_force_since`, so retrieval can filter by the reference date.
- The **knowledge horizon** is the last date each source type can speak for. A question dated
  after the horizon of its source type is refused before the LLM is called (the date gate).
- The rates horizon is 2026-01-31: rate amendments 19/2025 (in force 2026-02-01) and 01/2026
  (in force 2026-05-01) are in the raw data but not applied to the index rows.

### Eval set

| Split | Hand-written | Templated | Total |
|---|---|---|---|
| dev | 30 | 64 | 94 |
| test | 20 | 44 | 64 |
| **all** | **50** | **108** | **158** |

<!-- Source: eval/results/generation_dev_v4_items.jsonl and
eval/results/generation_test_final_judge-v2_items.jsonl (count of records by "source"). -->

- **Categories:** date-sensitive (the answer depends on which version was in force), numeric
  (rate lookups), direct lookup, interpretive (applying a provision to facts) and unanswerable
  (should abstain).
- **Templated items** are generated from provision versions: "what does rule X provide" on
  either side of an amendment date, and rate lookups.
- **Split:** 60/40 dev/test, stratified by source and category. Linked items (same
  `group_id`) stay in the same split, and the split is frozen.
- **Tuning:** all tuning (fusion weights, the abstention threshold, the judge, prompts) was
  done on dev.

### Models

| Role | Model |
|---|---|
| Generator | `gemini-3.5-flash-lite`, temperature 0, JSON claims with citation ids |
| Judge | Groq `openai/gpt-oss-120b`, temperature 0 |
| Embeddings | `bge-m3`, local |
| Reranker / confidence gate | `bge-reranker-v2-m3` cross-encoder, local |

<!-- Sources: generator_model and judge_model columns of eval/results/generation.csv.
Embedding and reranker models from the retrieval code (not an eval/results file). One dev v3
answer came from the fallback gemini-3.1-flash-lite (generation.csv generator_model on the
2026-11-10T07:47:09 row). -->

**Calculator tools:** `rate_lookup` and `gst_split` are deterministic Python functions the
generator can call in the API. Every evaluation in sections 2 to 5 ran with them off; section 8
reports them separately.

### Metrics

| Metric | Definition |
|---|---|
| Recall@5 / Recall@10 | Share of gold provisions among the top 5 / 10 retrieved records |
| MRR@10 | Mean reciprocal rank of the first gold record |
| Correct | Answerable item answered and judged `correct`, or unanswerable item abstained |
| Citation precision | Cited provision paths that are gold / all cited paths (answered items) |
| Citation recall | Gold paths cited / all gold paths (answerable items) |
| Abstention TP / FP / FN | Abstained on unanswerable / abstained on answerable (false abstention) / answered unanswerable |
| 95% CI | Percentile bootstrap, 1,000 resamples of the slice's items, fixed seed |

---

## 2. Retrieval ablation ladder (test)

Each rung adds one feature to the one above it. The ladder was run on test after the
rate-index fix, before the split BM25 index and typed citations existed.

| Config | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|
| naive (fixed-size chunks, dense) | 0.226 | 0.290 | 0.193 |
| dense (structured chunks) | 0.686 | 0.774 | 0.421 |
| bm25 | 0.540 | 0.621 | 0.392 |
| dense + bm25 (RRF fusion) | 0.669 | 0.766 | 0.469 |
| + exact match | 0.927 | 0.992 | 0.642 |
| **+ date filter** | **0.976** | **0.976** | **0.976** |
| + reranker | 0.976 | 0.976 | 0.984 |
| final config (+ split BM25, typed citations) | 0.976 | 0.976 | 0.984 |

<!-- Source: eval/results/retrieval.csv, split=test, source=all, category=overall. Ladder:
timestamp 2026-10-11T23:25:05+00:00 (index 3,777 rows; naive index 2,127 chunks). Final config
row: timestamp 2026-11-10T13:54:56+00:00, config +reranker,
fusion rrf(k=10,...,bm25=act_rules+rates,typed_citations). n = 62 test items with gold paths. -->

**Headline: the date filter.**

| Date-sensitive items (n=38) | Recall@10 | MRR@10 |
|---|---|---|
| + exact match | 1.000 | 0.572 |
| + date filter | 1.000 | 1.000 |

<!-- Source: eval/results/retrieval.csv, split=test, timestamp 2026-10-11T23:25:05+00:00,
source=all, category=date_sensitive. -->

- **Ranking, not finding:** without the date filter the right provision is usually found
  (Recall@10 1.000), but older and newer versions of it compete at the top. The filter keeps
  only the version valid on the reference date, so it lands at rank 1.
- **Exact match** (section and rule tokens such as `17(5)` or `rule 36(4)`) gives the largest
  Recall@10 jump: 0.766 to 0.992.
- **On the final config,** every test category is at Recall@10 1.000 except interpretive,
  at 0.625 (n=4).

---

## 3. Generation (test)

### v1 → v2 → final

| Run (judge) | Correct | Citation precision | Citation recall | Abstention TP / FP / FN |
|---|---|---|---|---|
| v1 (judge v1) | 53/64 = 0.828 [0.719, 0.906] | 57/58 [0.948, 1.000] | 57/64 = 0.891 [0.803, 0.968] | 2 / 5 / 0 |
| v2 (judge v1) | 52/64 = 0.812 [0.719, 0.906] | 60/61 [0.951, 1.000] | 60/64 = 0.938 [0.855, 1.000] | 2 / 2 / 0 |
| v2 (judge v2) | 53/64 = 0.828 [0.734, 0.922] | 60/61 [0.951, 1.000] | 60/64 = 0.938 [0.855, 1.000] | 2 / 2 / 0 |
| **final (judge v2)** | **53/64 = 0.828 [0.734, 0.922]** | **61/62 [0.952, 1.000]** | **61/64 = 0.953 [0.900, 1.000]** | **2 / 1 / 0** |
| final, shipped API config (judge v2) | 53/64 = 0.828 [0.734, 0.922] | 61/62 [0.952, 1.000] | 61/64 = 0.953 [0.900, 1.000] | 2 / 1 / 0 |

<!-- Source: eval/results/test_final_comparison.csv (source=all, category=overall), computed
from generation_test_items.jsonl (v1), generation_test_v2_items.jsonl and
generation_test_v2_judge-v2_items.jsonl (v2), generation_test_final_judge-v2_items.jsonl
(final) and generation_test_final_shipped_judge-v2_items.jsonl (shipped). Correct rates and
CIs match the corresponding rows of eval/results/generation.csv. -->

What changed between runs:
- **v2:** the rate index was rebuilt, one row per schedule entry.
- **final:** adds the date gate, the split BM25 index, typed citations and judge v2.

**Plain reading:**
- **No significant change overall:** correctness stayed at 53/64, and all three intervals
  overlap.
- **Rate questions** went from 13/16 to 16/16.
- **False abstentions** went from 5 to 1.
- **Citation recall** went from 0.89 to 0.95.
- **Citation precision** stayed at one wrong citation per run.

### Final run by source and category (test, judge v2)

| Slice | n | Correct | Citation precision | Citation recall | Abstention TP / FP / FN |
|---|---|---|---|---|---|
| hand, all | 20 | 16/20 = 0.80 [0.60, 0.95] | 17/18 | 17/20 | 2 / 1 / 0 |
| templated, all | 44 | 37/44 = 0.84 [0.73, 0.93] | 44/44 | 44/44 | 0 / 0 / 0 |
| date-sensitive | 38 | 31/38 = 0.82 [0.68, 0.92] | 38/38 | 38/39 | 0 / 0 / 0 |
| numeric (rates) | 16 | 16/16 = 1.00 | 16/16 | 16/16 | 0 / 0 / 0 |
| direct lookup | 4 | 3/4 = 0.75 [0.25, 1.00] | 4/5 | 4/4 | 0 / 0 / 0 |
| interpretive | 4 | 1/4 = 0.25 [0.00, 0.75] | 3/3 | 3/5 | 0 / 1 / 0 |
| unanswerable | 2 | 2/2 | n/a | n/a | 2 / 0 / 0 |

<!-- Source: eval/results/test_final_comparison.csv, run "final (judge v2)". -->

### Dev history (v1 → v4)

| Run | Change | Judge | Correct | Citation recall | Abstention TP / FP / FN |
|---|---|---|---|---|---|
| v1 | first pipeline | v1 | 66/94 = 0.702 [0.606, 0.787] | 89/92 | 3 / 2 / 0 |
| v2 | rate index: one row per schedule entry | v1 | 66/94 = 0.702 [0.606, 0.798] | 90/92 | 2 / 1 / 1 |
| v3 | date gate + split BM25 | v1 | 69/94 = 0.734 [0.628, 0.830] | 90/92 | 3 / 1 / 0 |
| v4 | rate horizon from unapplied amendments + typed citations | v1 | 69/94 = 0.734 [0.638, 0.830] | 90/92 | 3 / 1 / 0 |
| v4 | same answers | v2 | 71/94 = 0.755 [0.660, 0.840] | 90/92 | 3 / 1 / 0 |

<!-- Source: eval/results/generation.csv, split=dev, source=all, category=overall. v1: row
2026-10-11T22:41:00+00:00 (recomputed with CIs from generation_dev_items.jsonl); v2:
2026-11-10T06:19:14; v3: 2026-11-10T07:47:09; v4: 2026-11-10T11:34:01; v4 judge v2:
2026-11-10T12:49:55 (judge_version v2). -->

**Noise level on dev:** answer text changes when the tail of the context changes, so some
items flip between runs even when the gold provision leads the context in both.

| Pair of dev runs | Items whose correctness flipped | Gained | Lost |
|---|---|---|---|
| v1 → v2 | 10 | 5 | 5 |
| v2 → v3 | 7 | 5 | 2 |
| v3 → v4 | 2 | 1 | 1 |
| prompt baseline → provision-first | 9 | 4 | 5 |

<!-- Source: per-item files eval/results/generation_dev_items.jsonl,
generation_dev_v2_items.jsonl, generation_dev_v3_items.jsonl, generation_dev_v4_items.jsonl,
generation_dev_pv-baseline_judge-v2_items.jsonl, generation_dev_pv-provision_first_judge-v2_items.jsonl
(score.correct compared by item key). -->

Up to 5 dev items move each way between runs, so a net change of 3–4 items on 94 is within
noise. The dev confidence intervals are about ±0.09 wide.

---

## 4. Judge validation

| Design | |
|---|---|
| Rows | 34 dev answers from run v3 |
| Selection | every answer judge v1 called `incorrect` (3) or `partially_correct` (21), plus 10 random `correct` answers (fixed seed) |
| Human label | `human_verdict` filled in by hand for all 34 |

| Judge | Agreement with human | Cohen's kappa | Verdicts (correct / partial / incorrect) |
|---|---|---|---|
| v1 (gold answer only) | 29/34 | 0.72 | 10 / 21 / 3 |
| v2 (gold answer + gold provision text + rubric line) | 32/34 | 0.89 | 14 / 18 / 2 |

<!-- Source: eval/results/judge_audit_dev.csv (human_verdict, judge_verdict = v1) and
eval/results/judge_audit_dev_judge-v2.csv (human_verdict, judge_v1, judge_v2). Kappa is
unweighted Cohen's kappa over the three labels computed from these columns (0.721 and 0.891). -->

- **What v2 adds:** the text of each gold record in force on the reference date. For a rate
  schedule, that's the row the gold answer quotes. It also adds the rubric line "Extra
  statements supported by the provision text are not errors; omissions of material conditions
  are."
- **Where v2 disagrees:** both of its disagreements are rule 137 items it calls `correct`
  where the human said `partially_correct`.
- **Caveat:** v2 was designed from the failure modes seen in these same 34 rows (v1 marking
  supported extra detail as wrong), so its agreement on them is likely optimistic. No
  held-out human audit exists.

---

## 5. Prompt optimisation (dev, judge v2)

Four generator prompt variants were each run on all 94 dev items:

| Variant | Idea |
|---|---|
| (a) baseline | the original prompt |
| (b) strict citations | every claim must cite and quote the deciding words |
| (c) few-shot | two worked examples, on provisions and goods that appear in no eval item |
| (d) provision-first | identify the governing provision and its in-force dates before the claims |

### Screen without the judge

| Variant | Citation precision | Citation recall | Abstention TP / FP / FN | Format failures | Tokens in / out | Generator latency (mean) | Screen |
|---|---|---|---|---|---|---|---|
| baseline | 1.000 | 0.978 | 3 / 1 / 0 | 0 | 2,700 / 377 | 6.3 s | baseline |
| strict citations | 1.000 | 0.967 | 3 / 2 / 0 | 0 | 2,755 / 383 | 5.3 s | dropped: lower recall, more false abstentions |
| few-shot | 1.000 | 0.978 | 3 / 1 / 0 | 0 | 3,526 / 280 | 4.2 s | survives (+24% tokens) |
| provision-first | 1.000 | 0.978 | 3 / 1 / 0 | 0 | 2,769 / 423 | 6.5 s | survives; judged (tie, fewer tokens) |

<!-- Source: eval/results/prompt_variants_prescreen_dev.csv. Baseline latency from an
uncached timing pass (eval/results/prompt_variants_latency_dev.json); other variants from the
recorded call latency. 91 of 94 items reached the generator. -->

- **Drop rule:** a variant was dropped if it was worse than the baseline on citation recall
  or false abstentions, or used more than 30% more tokens without a citation gain.
- **Tie-break:** only one challenger was judged, to keep judge cost down. Few-shot and
  provision-first tied on every quality metric, and provision-first won on tokens.

### Finalist comparison

| Variant | Correct | Citation P / R | Abstention F1 | Partially correct | Tokens in / out |
|---|---|---|---|---|---|
| **baseline** | **71/94 = 0.755 [0.660, 0.840]** | 1.000 / 0.978 | 0.857 | 19 | 2,700 / 377 |
| provision-first | 70/94 = 0.745 [0.649, 0.830] | 1.000 / 0.978 | 0.857 | 20 | 2,769 / 423 |

<!-- Source: eval/results/prompt_variants_dev.csv. -->

**Decision: the baseline prompt was kept.** Provision-first scored one item lower, which is
well inside the baseline's interval, and it costs more tokens. No variant showed a
significant gain.

---

## 6. Error analysis: what was found and fixed

| Problem | How it showed up | Fix | Measured effect |
|---|---|---|---|
| Rate-index defect: layout text extraction scrambled the columns of many schedule rows, so goods descriptions and rates did not match | a false abstention on a rate question whose row could not be found | Re-parsed the schedules with `pdftotext -raw`, one index row per schedule entry, plus a test that every parsed description is verbatim in exactly one row | index 1,375 → 3,777 rows; test rate questions 13/16 → 16/16; test dense MRR on numeric items 0.568 → 0.750 |
| BM25 side effect: the 2,723 short rate rows changed term statistics for the long Act and Rules rows | dev bm25 date-sensitive Recall@10 fell 0.554 → 0.429 after the rate fix | Split BM25: Act/Rules and rate rows in separate indexes, merged in RRF | dev dense+bm25 overall MRR 0.458 → 0.530; full-pipeline retrieval unchanged |
| Future-dated rate question answered from the current schedule | "rate on Portland cement in 2028" answered instead of refused | Date gate: abstain when the reference date is after the source type's knowledge horizon | dev abstention TP / FP / FN 2 / 1 / 1 → 3 / 1 / 0 |
| Rate rows pulled into rule-question contexts (e.g. "rule 137" matching S. No. 137) | 8 dev rule/section questions had rate rows in context | Typed citations: a question citing rules or sections skips the rate BM25 index (and vice versa) | 8 → 0 such contexts; retrieval metrics unchanged |
| Rule 89 refusal | the model abstains on `ds-r89-10B@2021-12-31` ("Is a refund application under rule 89(1) subject to the provisions of rule 10B?") in every dev run | not fixed | the one remaining dev false abstention |

<!-- Sources: index sizes from eval/results/retrieval.csv (index_size 1,375 on the
2026-10-11T22:48:44 test run, 3,777 from 2026-10-11T23:25:05). Test numeric dense MRR 0.5677 →
0.75 from the same two runs (config dense, category numeric). Rate questions 13/16 → 16/16
from eval/results/test_final_comparison.csv (category numeric, v1 vs final). Dev bm25
date-sensitive Recall@10: 0.554 (dev run 2026-10-11T19:28:39) → 0.429 (2026-10-11T22:59:23) →
0.464 split (2026-11-10T07:05:14), and dense+bm25 overall MRR 0.458 → 0.530, in
eval/results/retrieval.csv. The 2,723 rate-row count is 3,777 minus the 1,054 Act/Rules rows
(pipeline docstring, not a results file). Abstention counts from eval/results/generation.csv
dev v2 and v3 overall rows. "8 → 0" rate rows in non-rate dev contexts: counted from the
context_ids of generation_dev_v3_items.jsonl and generation_dev_v4_items.jsonl. Rule 89 item
from generation_dev_items.jsonl, _v2, _v3 and _v4_judge-v2 (abstain_reason "model",
answerable true). -->

- **Split BM25 trade-off:** in the BM25-only rung the split costs some recall (dev
  date-sensitive Recall@10 0.464), because the two lists are interleaved. The fused rungs
  improved.
- **Not quantified here:** what share of schedule rows the old extraction scrambled. That
  was measured during the investigation but is not in a committed result file.

---

## 7. Guardrails and observability

### Input guardrails (30-question guardrail set)

Three input guardrails sit in front of the API: question length, prompt injection, and out
of scope (by keywords, then by retrieval confidence when the keywords are inconclusive).

| Configuration | Refused | Should refuse | Refusal precision | Refusal recall | Out-of-scope recall | Injection recall |
|---|---|---|---|---|---|---|
| keywords only | 16 | 20 | 1.000 | 0.800 | 0.70 | 0.90 |
| keywords + confidence | 22 | 20 | 0.909 | 1.000 | 1.00 | 0.90 |

<!-- Source: eval/results/guardrails.csv. -->

The guardrail set was scored once. Its patterns were not tuned on it.

### Shipped API config

- **What ships:** the API runs the final config plus the input guardrails and the output
  guardrail `strict_claims`. With `strict_claims`, the answer abstains if any generated
  claim cites no retrieved record.
- **Why a replay:** neither guardrail is wired into the eval harness, so the shipped-config
  numbers were obtained by replaying them on the final test answers with no new LLM calls.
- **How the replay works:** the input guardrails were run on each question, using the run's
  recorded confidence for the scope check. Each stored generator output was then passed back
  through the answer pipeline with `strict_claims` on.
- **Checked on dev first:** with `strict_claims` off, the replay reproduced every dev answer
  exactly.
- **Result on test:** 0 outcomes changed (section 3 table, "shipped API config" row).

### Latency

| Measure | Value |
|---|---|
| Generator call, baseline prompt (mean of 91 uncached dev calls) | 6.3 s |
| Generator call, other variants (mean) | 4.2 s to 6.5 s |

<!-- Source: eval/results/prompt_variants_prescreen_dev.csv (mean_latency_s). -->

- **Not in the committed results:** request-level latency per stage (retrieve, rerank,
  generate) and cold start. The API writes them to `logs/requests.jsonl`, which is
  git-ignored.
- **Cold start:** the first request loads the embedding and cross-encoder models; the README
  notes it takes 30 seconds or more.

### Cost (estimates)

| Item | Value |
|---|---|
| Judge cost, all logged steps since the Groq paid upgrade | $0.106 (256 judge calls) |
| Judge cost per call (final test run) | about $0.0003 |
| Generator tokens per request (baseline, mean) | 2,700 input / 377 output |
| Generator cost per request | $0 on the Gemini free tier; about $0.0018 at list price |

<!-- Sources: eval/results/cost_log.csv (sum of groq_cost_usd = 0.1056 over 9 steps, sum of
groq_calls = 256; final test step 61 calls, $0.0207). Prices are estimates, not bills: Groq
gpt-oss-120b $0.15 / $0.75 per million input / output tokens (eval/cost.py); Gemini list price
$0.30 / $2.50 per million (observability/config.py), applied to the mean tokens in
eval/results/prompt_variants_dev.csv (baseline): 2,700 x 0.30e-6 + 377 x 2.50e-6 = $0.0018.
The first cost_log step (audit re-judge) may include calls made before the upgrade. -->

---

## 8. Calculator tools

Calculator tools (rate_lookup, gst_split) were added after the frozen evaluation. They are off
in the eval harness and on in the API. The shipped-config replay on test (0/64 changed) covered
input guardrails, strict_claims and the number check; tool selection was not replayed on test.

### Calculation set (n=15, through the API)

The 15 questions were written for the tools, with expected outputs worked out by hand. They are
separate from the frozen main eval, which ran with the tools off.

| Metric | Result |
|---|---|
| Tool selection (the expected tools, in order) | 14/15 = 0.933 |
| Arguments (over items that expect a tool) | 13/14 = 0.929 |
| Final numbers: every expected number appears in the answer | 5/12 = 0.417 |
| Outcome (answered or abstained as expected) | 13/15 = 0.867 |

<!-- Source: eval/results/calc.csv (run 2026-11-11T08:09:50+00:00) and
eval/results/calc_items.jsonl; questions and hand calculations in eval/questions_calc.jsonl. -->

Failures:

| Item | What happened |
|---|---|
| calc-04 | The question stated the rate (5%), but the planner called `rate_lookup` with the query "GST 5%" instead of `gst_split`. The lookup was ambiguous and the answer abstained. |
| calc-05 | `gst_split` was called correctly, but the generator abstained ("model") instead of quoting the tool output. |
| calc-01, 02, 03, 06, 13 | Each answer gave the amounts the question asked for, correctly, but left out totals that the expected list also included (for example the invoice total). |

No answer contained a number that was not in a tool output.

### Routing on the 24 dev rate questions (tools on, no judge)

| Measure | Result |
|---|---|
| Routed to `rate_lookup` | 24/24 |
| Lookup returned a single row | 13/24 |
| That row's rate equals the gold rate | 12/13 |
| Answer states the gold rate | 13/24 (dev v4, tools off: 24/24 correct) |
| Items whose outcome differs from dev v4 | 11 |

<!-- Source: eval/results/tool_routing_dev.csv (run 2026-11-11T08:22:43+00:00) and
eval/results/tool_routing_dev.jsonl; dev v4 from generation_dev_v4_judge-v2_items.jsonl. -->

Why the 11 differ:
- **9 failed calls:** the templated questions quote both the goods description and the tariff
  code, and the planner passed both. `rate_lookup` accepts exactly one of them, so it rejected
  the call.
- **2 ambiguous lookups:** "air-conditioning machines" matches more than one schedule row, on
  both dates.
- **1 wrong row, not counted among the 11:** for tractors on 2025-10-01, word matching picked
  the "road tractors for semi-trailers" row (9%), because the words of the exclusion "other than
  road tractors…" also matched. The answer still stated the gold 2.5% by citing the retrieved
  schedule row.

**Plain reading:** with the tools on, the API answers the dev rate questions worse than the
frozen pipeline without tools. The rate questions should not be routed through `rate_lookup`
until it accepts a description and a tariff code together and handles exclusions.

### Limits of the tools

- **Rounding:** amounts use Decimal arithmetic rounded half up to 2 decimals. The tax is
  rounded once on the combined rate, and CGST takes the odd paisa. This is an assumption; the
  law is not the source of it.
- **Combined rate:** `combined_rate` is twice the central rate, which assumes the State rate
  equals the central rate. The State GST Acts and the IGST rate notifications are not indexed.
- **Ambiguous matches:** a lookup that matches more than one row returns every candidate and no
  rate, and the answer abstains.
- **No citation for `gst_split`:** the IGST Act and the State GST Acts are not in the index,
  and CGST section 9(1) is only an unverified snapshot row there, so the split cites nothing and
  says so in a note.

---

## 9. Limitations and next steps

| Limitation | Detail |
|---|---|
| Interpretive questions are the weakest | 1/4 correct on test, retrieval Recall@10 0.625. With n=4 the estimate is very uncertain. |
| Long-rule summaries are often partial | Templated "what does rule X provide" items for long rules (89, 43, 60) are mostly judged `partially_correct`. The citations are right; the answers summarise. |
| GST only | v1 covers the CGST Act, CGST Rules and central tax rate notifications; income tax is planned for v2. |
| Undated Rules amendments | 6 Rules amendment instructions take effect "from a date to be notified" (notifications 07/2025 and 35/2021). They cannot be placed on the timeline, so they are not applied; 2 more (45/2023) were superseded by 51/2023. |
| Rate amendments not applied | 19/2025 and 01/2026 amend schedule 9/2025 but are not applied to the index rows, which is why the rates horizon is 2026-01-31. |
| Small hand-written test set | 20 hand-written test items. The templated items (44) are easier and more uniform. |
| Test split used three times | v1, v2 and final were all run on test; there was no tuning on test. |
| LLM judge | Correctness depends on an LLM judge validated on 34 dev rows, and the design of v2 was informed by those same rows. |
| Guardrails replayed, not wired in | The shipped-config numbers come from a replay; the eval harness itself does not run the guardrails. |

<!-- Sources: interpretive figures from eval/results/test_final_comparison.csv and
eval/results/retrieval.csv (test, +reranker, interpretive); long-rule verdicts from
eval/results/generation_test_final_judge-v2_items.jsonl; undated amendments from
data/processed/rule_version_report.json (pending_commencement: 8 entries, 6
"awaiting_notification", 2 "superseded by 51/2023"), not an eval/results file. -->

**Next steps**

- Wire the input guardrails and `strict_claims` into the eval harness so the shipped config
  is measured directly.
- Apply rate amendments 19/2025 and 01/2026 to the index rows, which moves the rates horizon
  forward.
- Write more hand-written interpretive questions, and a fresh, untouched test set.
- Run a second human audit on rows not used to design judge v2.
