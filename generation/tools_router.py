"""Calculator routing (GenerationConfig.tools): the LLM picks a tool and its arguments,
Python runs it, and the answer quotes the tool's numbers.

1. Plan: a Gemini function-calling turn with the tool declarations below. The model may call
   tools over up to MAX_ROUNDS turns; each result is sent back to it, so a rate found by
   rate_lookup can be passed to gst_split. Planning stops after a round of gst_split calls or
   a failed or inconclusive call (no further turn could change the outcome). The reference
   date is never an argument: Python passes it.
2. Run: tools.rate_lookup / tools.gst_split, pure functions; an invalid argument is recorded
   as a failed call.
3. Each tool result becomes a context chunk (id ``TOOL-<n>``, path ``TOOL/<tool>``) for the
   answer call, which must copy the tool's numbers and cite the chunk.
4. Number check: every number in the answer's claims and assumptions must appear in a tool
   output, in the text of a cited chunk or in the question; otherwise the answer abstains.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from generation.context import Chunk
from generation.timing import ms
from retrieval.store import OPEN_END
from tools.gst_split import gst_split
from tools.rate_lookup import rate_lookup
from tools.result import ToolResult

MAX_ROUNDS = 3
MAX_CALLS = 6
TOOL_ID_PREFIX = "TOOL-"
NO_TOOL = "NO_TOOL"

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "rate_lookup",
        "description": (
            "Central tax rate of goods on the reference date, from the indexed rate schedules. "
            "Give exactly one of query (the goods as described in the question) or hsn (an HSN "
            "or tariff code). Returns rate (central tax, per cent) and combined_rate (CGST + "
            "SGST, or IGST), or every candidate row when more than one matches."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "description of the goods"},
                "hsn": {"type": "STRING", "description": "HSN / tariff code, digits"},
            },
        },
    },
    {
        "name": "gst_split",
        "description": (
            "Tax amounts for a supply: CGST and SGST for an intra-State supply, IGST for an "
            "inter-State supply, at a combined GST rate in per cent. With inclusive=true the "
            "amount is a tax-inclusive price and the tax is backed out of it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "taxable_value": {
                    "type": "STRING",
                    "description": "amount in rupees exactly as stated, e.g. 10000 or 1180.50",
                },
                "rate": {
                    "type": "STRING",
                    "description": "combined GST rate in per cent, e.g. 18 (copy combined_rate "
                    "from rate_lookup)",
                },
                "supply": {"type": "STRING", "enum": ["intra", "inter"]},
                "inclusive": {"type": "BOOLEAN"},
            },
            "required": ["taxable_value", "rate", "supply"],
        },
    },
]

PLANNER_SYSTEM = """\
You route questions about Indian GST to calculator tools. You never compute a number.
- rate_lookup: the central tax rate of goods on the reference date (by description or HSN).
- gst_split: CGST/SGST (intra-State) or IGST (inter-State) amounts for a taxable value, or for \
a tax-inclusive price with inclusive=true, at a combined GST rate.
Call a tool only when the question asks for the rate of goods or for tax amounts. Copy every \
argument from the question or from an earlier tool result exactly; when the rate comes from \
rate_lookup, pass its combined_rate to gst_split. A supply within one State is intra; between \
two States, or into or out of India, is inter. If rate_lookup returns candidates, a refusal or \
nothing, do not call gst_split. If no tool is needed, reply with the single word NO_TOOL.
"""

TOOL_ANSWER_RULES = """\
- Context chunks whose ID starts with TOOL- are calculator outputs for this question. Copy \
every amount, rate and figure from them exactly as written and cite that TOOL- ID; never \
compute, convert or round a number yourself.
"""

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    ok: bool
    latency_ms: float
    output: dict[str, Any] | None = None  # ToolResult.model_dump()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolRun:
    calls: list[ToolCall] = field(default_factory=list)
    llm: list[dict[str, Any]] = field(default_factory=list)  # one entry per planner turn
    plan_ms: float = 0.0
    tools_ms: float = 0.0


def provisions_by_path(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["path"]].append(row)
    return dict(out)


def execute(
    name: str,
    args: Mapping[str, Any],
    ref_date: date,
    provisions: Mapping[str, Sequence[dict]] | None = None,
) -> ToolCall:
    """Run one tool call; a bad argument or an unknown tool is a failed call, not a crash."""
    started = time.perf_counter()
    arguments = dict(args)
    try:
        if name == "rate_lookup":
            query, hsn = arguments.get("query"), arguments.get("hsn")
            result = rate_lookup(query or None, hsn=hsn or None, on=ref_date)
        elif name == "gst_split":
            result = gst_split(
                arguments["taxable_value"],
                arguments["rate"],
                arguments["supply"],
                inclusive=bool(arguments.get("inclusive", False)),
                on=ref_date,
                provisions=provisions,
            )
        else:
            raise ValueError(f"unknown tool {name!r}")
    except (ValueError, KeyError, TypeError) as error:
        return ToolCall(name, arguments, False, ms(time.perf_counter() - started), error=str(error))
    return ToolCall(name, arguments, True, ms(time.perf_counter() - started), result.model_dump())


def plan_and_run(
    generator,
    question: str,
    ref_date: date,
    provisions: Mapping[str, Sequence[dict]] | None = None,
) -> ToolRun:
    """Planner turns and tool executions; ``calls`` is empty when the model calls no tool."""
    run = ToolRun()
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [{"text": f"Reference date: {ref_date.isoformat()}\nQuestion: {question}"}],
        }
    ]
    for _ in range(MAX_ROUNDS):
        started = time.perf_counter()
        turn = generator.complete_tools(PLANNER_SYSTEM, contents, TOOL_DECLARATIONS)
        run.plan_ms += ms(time.perf_counter() - started)
        run.llm.append(
            {
                "provider": turn.provider,
                "model": turn.model,
                "cached": turn.cached,
                "usage": turn.usage,
                "latency_s": turn.latency_s,
            }
        )
        calls = turn.function_calls
        if not calls or len(run.calls) >= MAX_CALLS:
            break
        responses = []
        for call in calls[: MAX_CALLS - len(run.calls)]:
            started = time.perf_counter()
            done = execute(call.get("name", ""), call.get("args") or {}, ref_date, provisions)
            run.tools_ms += ms(time.perf_counter() - started)
            run.calls.append(done)
            output = done.output if done.ok else {"error": done.error}
            responses.append(
                {"functionResponse": {"name": done.name, "response": {"output": output}}}
            )
        # Nothing can follow gst_split, and a failed or inconclusive call ends in an
        # abstention, so another planner turn would only cost a call.
        latest = run.calls[-len(responses) :]
        if all(c.name == "gst_split" for c in latest) or any(
            not c.ok or c.output["status"] != "ok" for c in latest
        ):
            break
        contents = [
            *contents,
            {"role": "model", "parts": turn.parts},
            {"role": "user", "parts": responses},
        ]
    return run


def tool_chunks(calls: Sequence[ToolCall], ref_date: date) -> list[Chunk]:
    """Context chunks for the answer call, one per successful tool call."""
    chunks = []
    for number, call in enumerate((c for c in calls if c.ok), 1):
        result = ToolResult.model_validate(call.output)
        arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
        text = f"Arguments: {arguments}\n{result.text()}"
        chunk_id = f"{TOOL_ID_PREFIX}{number}"
        chunks.append(
            Chunk(
                chunk_id,
                chunk_id,
                f"TOOL/{call.name}",
                "tool",
                ref_date.isoformat(),
                OPEN_END,
                text,
                1.0,
                False,
            )
        )
    return chunks


def chunks_from_records(records: Sequence[dict[str, Any]], ref_date: date) -> list[Chunk]:
    """tool_chunks for recorded calls (AnswerResult.tool_calls); the ids match the answer's."""
    return tool_chunks([ToolCall(**record) for record in records], ref_date)


def numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def _value(number: str) -> Decimal | None:
    try:
        return Decimal(number.replace(",", "")).normalize()
    except InvalidOperation:
        return None


def unsupported_numbers(answer_texts: Sequence[str], source_texts: Sequence[str]) -> list[str]:
    """Numbers in the answer whose value appears in none of the source texts."""
    allowed = {_value(n) for text in source_texts for n in numbers(text)}
    missing = []
    for text in answer_texts:
        for number in numbers(text):
            if _value(number) not in allowed:
                missing.append(number)
    return list(dict.fromkeys(missing))
