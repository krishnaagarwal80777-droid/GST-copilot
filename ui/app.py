"""Streamlit UI over the API.

    uvicorn api.app:app --port 8000      # in one terminal
    streamlit run ui/app.py              # in another (GST_API_URL overrides the API address)

The Observability tab reads the request log (observability.config.LOG_PATH) directly.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit puts only ui/ on the path
    sys.path.insert(0, str(ROOT))

from observability.config import LOG_PATH  # noqa: E402
from observability.report import load, summarize  # noqa: E402
from ui.client import (  # noqa: E402
    API_URL,
    ask,
    calculation_markdown,
    citation_title,
    claims_markdown,
)

st.set_page_config(page_title="GST Tax Copilot", layout="wide")
st.title("GST Tax Copilot")
st.caption(
    "Answers about Indian GST law as on a reference date, with citations to the provisions "
    "in force on that date. Not professional tax or legal advice."
)

ask_tab, obs_tab = st.tabs(["Ask", "Observability"])

with ask_tab:
    with st.form("ask"):
        question = st.text_area("Question", height=100, max_chars=2000)
        left, right = st.columns(2)
        reference_date = left.date_input(
            "Reference date (the law as on)", value=date.today(), min_value=date(2017, 7, 1)
        )
        use_knowledge = right.checkbox(
            "Set a knowledge date", help="Only use amendments enacted by this date."
        )
        knowledge_date = right.date_input(
            "Knowledge date", value=date.today(), max_value=date.today()
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving provisions and drafting the answer..."):
                result = ask(question, reference_date, knowledge_date if use_knowledge else None)
            if result.error:
                st.error(result.error)
            else:
                data = result.data
                if data["guardrail"]:
                    st.warning(data["answer"])
                elif data["abstained"]:
                    st.info(
                        "**No answer given.** The assistant abstains rather than guess when "
                        "the provisions in force on the reference date do not clearly answer "
                        f"the question (reason: `{data['abstain_reason']}`)."
                    )
                    st.markdown(data["answer"])
                else:
                    st.markdown(claims_markdown(data))
                    st.markdown(f"**Reference date used:** {data['reference_date']}")
                    st.subheader("Citations")
                    for citation in data["citations"]:
                        with st.expander(citation_title(citation)):
                            st.markdown(f"`{citation['version_id']}`  ·  {citation['doc_type']}")
                            st.text(citation["text"])
                            if citation["source_url"]:
                                st.markdown(
                                    f"Source: [{citation['source_file']}]({citation['source_url']})"
                                )
                            elif citation["source_file"]:
                                st.markdown(f"Source file: `{citation['source_file']}`")
                if data.get("tool_fallback"):
                    st.caption(
                        "The rate lookup did not find a single schedule row, so this answer "
                        "comes from the retrieved provisions."
                    )
                if data.get("tool_calls"):
                    with st.expander("Calculation", expanded=True):
                        st.caption(
                            "Numbers computed by Python tools (the model only chose the tool "
                            "and its arguments)."
                        )
                        st.markdown(calculation_markdown(data))
                timings = "  ·  ".join(f"{k} {v:.0f} ms" for k, v in data["timings_ms"].items())
                model = data["model"] or "none (no LLM call)"
                st.caption(f"request {data['request_id']}  ·  model {model}  ·  {timings}")

with obs_tab:
    st.caption(f"From {LOG_PATH} (API at {API_URL})")
    st.button("Refresh")
    records = load(LOG_PATH)
    if not records:
        st.info("No requests logged yet.")
    else:
        summary = summarize(records)
        cols = st.columns(5)
        cols[0].metric("Requests", summary["requests"], help=f"errors: {summary['errors']}")

        def pct(value: float | None) -> str:
            return "-" if value is None else f"{value:.1%}"

        cols[1].metric("Guardrail rate", pct(summary["guardrail_rate"]))
        cols[2].metric("Abstention rate", pct(summary["abstention_rate"]))
        cols[3].metric("Cache hit rate", pct(summary["cache_hit_rate"]))
        cost = summary["mean_cost_usd"]
        cols[4].metric("Cost / request", "-" if cost is None else f"${cost:.5f}")
        st.subheader("Latency per stage")
        st.dataframe(
            [{"stage": stage, **row} for stage, row in summary["latency"].items()],
            hide_index=True,
        )
        left, right = st.columns(2)
        left.markdown("**Guardrails triggered**")
        left.json(summary["guardrails"] or {})
        right.markdown("**Abstention reasons**")
        right.json(summary["abstain_reasons"] or {})
        st.markdown(
            f"Tool calls: {summary['tool_calls'] or '{}'} in {summary['tool_requests']} "
            f"requests  ·  success rate {pct(summary['tool_success_rate'])}  ·  p50 latency "
            f"{summary['tool_latency_p50_ms'] or '-'} ms"
        )
        st.markdown(
            f"Generator calls: {summary['generator_calls']}  ·  mean tokens per call: "
            f"input {summary['mean_input_tokens']}, output {summary['mean_output_tokens']}  ·  "
            f"total cost ${summary['total_cost_usd']} (list prices; unpriced requests: "
            f"{summary['unpriced_requests']})"
        )
