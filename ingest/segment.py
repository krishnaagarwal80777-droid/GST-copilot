from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

SegmentType = Literal["section", "subsection", "clause", "sub_clause", "proviso", "explanation"]


class SegmentNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    node_type: SegmentType
    marker: str
    text: str = ""
    heading: str | None = None
    parent_path: str | None = None


SECTION = re.compile(r"^(?:Section\s+)?(?P<number>\d+[A-Z]?)(?:\s*[.:])\s*(?P<heading>.*)$", re.IGNORECASE)
SENTINEL_SECTION = re.compile(r"^Section\s+(?P<number>[A-Z][A-Z0-9_-]*)(?:\.?\s*)(?P<heading>.*)$")
SUBSECTION = re.compile(r"^\((?P<number>\d+[A-Z]?)\)\s*(?P<body>.*)$")
CLAUSE = re.compile(r"^\((?P<marker>[a-z])\)\s*(?P<body>.*)$")
SUB_CLAUSE = re.compile(r"^\((?P<marker>[ivxlcdm]+)\)\s*(?P<body>.*)$", re.IGNORECASE)
PROVISO = re.compile(r"^(Provided(?:\s+further)?\s+that\b[.\s—-]*)(.*)$", re.IGNORECASE)
EXPLANATION = re.compile(r"^(Explanation(?:\s+\d+)?\s*[.\-—:]\s*)(.*)$", re.IGNORECASE)


def _append_text(node: SegmentNode, text: str) -> None:
    cleaned = text.strip()
    if cleaned:
        node.text = f"{node.text} {cleaned}".strip()


def parse_act_text(text: str, act_id: str = "CGST") -> list[SegmentNode]:
    """Split line-structured Act text into hierarchical nodes with canonical paths.

    Structural markers must begin a line. Wrapped lines are appended to the active node.
    Section identifiers that are not numeric are accepted for clearly synthetic parser tests.
    """
    nodes: list[SegmentNode] = []
    structural_stack: list[SegmentNode] = []
    current: SegmentNode | None = None
    child_counts: dict[tuple[str, str], int] = defaultdict(int)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = SECTION.match(line) or SENTINEL_SECTION.match(line)
        if section_match:
            marker = section_match.group("number")
            heading = (section_match.group("heading") or "").strip()
            section = SegmentNode(
                path=f"{act_id}/s{marker}",
                node_type="section",
                marker=marker,
                heading=heading or None,
            )
            nodes.append(section)
            structural_stack = [section]
            current = section
            continue

        if not structural_stack:
            continue

        match = SUBSECTION.match(line)
        if match:
            marker, body = match.groups()
            while len(structural_stack) > 1:
                structural_stack.pop()
            parent = structural_stack[-1]
            node = SegmentNode(
                path=f"{parent.path}/ss{marker}",
                node_type="subsection",
                marker=marker,
                text=body,
                parent_path=parent.path,
            )
            nodes.append(node)
            structural_stack.append(node)
            current = node
            continue

        expected_parent = structural_stack[-1]
        sub_clause_match = SUB_CLAUSE.match(line)
        if expected_parent.node_type in {"clause", "sub_clause"} and sub_clause_match:
            while structural_stack[-1].node_type != "clause":
                structural_stack.pop()
            marker, body = sub_clause_match.groups()
            marker = marker.lower()
            parent = structural_stack[-1]
            node = SegmentNode(
                path=f"{parent.path}/scl({marker})",
                node_type="sub_clause",
                marker=marker,
                text=body,
                parent_path=parent.path,
            )
            nodes.append(node)
            structural_stack.append(node)
            current = node
            continue

        clause_match = CLAUSE.match(line)
        if clause_match:
            while structural_stack[-1].node_type not in {"section", "subsection"}:
                structural_stack.pop()
            marker, body = clause_match.groups()
            parent = structural_stack[-1]
            node = SegmentNode(
                path=f"{parent.path}/cl({marker})",
                node_type="clause",
                marker=marker,
                text=body,
                parent_path=parent.path,
            )
            nodes.append(node)
            structural_stack.append(node)
            current = node
            continue

        parent = structural_stack[-1]
        auxiliary_match = PROVISO.match(line)
        auxiliary_type: Literal["proviso", "explanation"] | None = None
        auxiliary_body = ""
        if auxiliary_match:
            marker, auxiliary_body = auxiliary_match.groups()
            auxiliary_type = "proviso"
        else:
            auxiliary_match = EXPLANATION.match(line)
            if auxiliary_match:
                marker, auxiliary_body = auxiliary_match.groups()
                auxiliary_type = "explanation"

        if auxiliary_type is not None:
            key = (parent.path, auxiliary_type)
            child_counts[key] += 1
            ordinal = child_counts[key]
            node = SegmentNode(
                path=f"{parent.path}/{auxiliary_type}{ordinal}",
                node_type=auxiliary_type,
                marker=marker.strip(),
                text=auxiliary_body,
                parent_path=parent.path,
            )
            nodes.append(node)
            current = node
            continue

        if current is not None:
            _append_text(current, line)

    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment line-structured Act text.")
    parser.add_argument("input", type=Path, help="Plain-text Act source")
    parser.add_argument("--act-id", default="CGST", help="Canonical act identifier")
    parser.add_argument("--section", action="append", help="Section number(s) to print")
    arguments = parser.parse_args()

    nodes = parse_act_text(arguments.input.read_text(encoding="utf-8"), arguments.act_id)
    if arguments.section:
        wanted = set(arguments.section)
        selected_sections = {
            node.path for node in nodes if node.node_type == "section" and node.marker in wanted
        }
        nodes = [
            node
            for node in nodes
            if node.path in selected_sections
            or any(node.path.startswith(f"{path}/") for path in selected_sections)
        ]
    print(json.dumps([node.model_dump() for node in nodes], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
