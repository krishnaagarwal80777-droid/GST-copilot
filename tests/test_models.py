from datetime import date

import pytest
from pydantic import ValidationError

from gst_copilot.models import EvalItem, GraphEdge, ProvisionRecord


def test_provision_record_requires_bitemporal_dates() -> None:
    record = ProvisionRecord(
        id="CGST/sTEST@v1",
        doc_type="act",
        act_id="CGST2017",
        path="CGST/sTEST",
        node_type="section",
        text="synthetic record",
        effective_from=date(2020, 1, 1),
        effective_to=None,
        in_force_since=date(2020, 1, 1),
    )

    assert record.effective_from == date(2020, 1, 1)
    assert record.effective_to is None
    assert record.in_force_since == date(2020, 1, 1)


def test_provision_record_rejects_inverted_effective_period() -> None:
    with pytest.raises(ValidationError):
        ProvisionRecord(
            id="CGST/sTEST@v1",
            doc_type="act",
            act_id="CGST2017",
            path="CGST/sTEST",
            node_type="section",
            text="synthetic record",
            effective_from=date(2020, 1, 2),
            effective_to=date(2020, 1, 1),
            in_force_since=date(2020, 1, 1),
        )


def test_graph_edge_and_eval_item_validate() -> None:
    edge = GraphEdge(
        source_id="CGST/sTEST@v1",
        target_id="CGST/sSAMPLE@v1",
        edge_type="CHILD_OF",
    )
    item = EvalItem(
        question="synthetic question",
        ref_date=date(2026, 1, 1),
        gold_paths=["CGST/sTEST"],
        gold_answer="synthetic answer",
        category="handwritten",
        answerable=True,
        source="hand",
    )

    assert edge.edge_type == "CHILD_OF"
    assert item.gold_paths == ["CGST/sTEST"]
