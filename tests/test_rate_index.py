from __future__ import annotations

from collections import defaultdict

import pytest

from ingest.rate_schedule import parse_rows
from retrieval.corpus import RATE_NOTIFICATIONS, rate_rows


@pytest.fixture(scope="module")
def index_rows():
    return rate_rows()


@pytest.mark.parametrize("notification", RATE_NOTIFICATIONS, ids=lambda n: n.number)
def test_every_parsed_row_is_verbatim_in_exactly_one_index_row(notification, index_rows) -> None:
    """Each schedule entry is one index row, so its description appears verbatim in exactly one
    row of its notification. A description that is repeated (or contained in another entry's
    description) appears once per entry that contains it, and in no other row."""
    slug = notification.number.replace("/", "-")
    texts = [row.text for row in index_rows if row.path.split("/")[1] == slug]
    parsed = parse_rows(notification)
    assert len(texts) == len(parsed)
    containing = defaultdict(int)
    for row in parsed:
        containing[row.description] = sum(
            1 for other in parsed if row.description in other.description
        )
    for row in parsed:
        hits = [text for text in texts if row.description in text]
        assert len(hits) == containing[row.description], (notification.number, row.s_no)
        own = f"S. No. {row.s_no}.{' ' + row.tariff if row.tariff else ''} {row.description}"
        assert sum(1 for text in hits if text.endswith(own)) == 1, (notification.number, row.s_no)


def test_index_rows_carry_rate_and_record_of_their_schedule(index_rows) -> None:
    row = next(r for r in index_rows if r.id == "CTR/9-2025/schII@v1#424")
    assert row.record_id == "CTR/9-2025/schII@v1" and row.path == "CTR/9-2025/schII"
    assert row.text.startswith(
        "Notification No. 9/2025-Central Tax (Rate), Schedule II - central tax rate 9 per cent: "
        "S. No. 424. 8437 Machines for cleaning, sorting or grading, seed, grain"
    )
    exempt = next(r for r in index_rows if r.id == "CTR/10-2025@v1#1")
    assert exempt.text.endswith("(nil central tax): S. No. 1. 0101 Live asses, mules and hinnies")
