"""Calculator tools: gst_split rounding, inclusive back-calculation, intra vs inter, and
rate_lookup matching, ambiguity and the horizon refusal. Synthetic rate rows are
placeholders (not legal text); one test reads the real index rows from data/raw."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools.gst_split import gst_split
from tools.rate_lookup import RateEntry, rate_lookup, words

D = Decimal


# --- gst_split ---------------------------------------------------------------------------


def test_intra_state_splits_the_rate_equally() -> None:
    out = gst_split(D("10000"), D("18"), "intra")
    assert out.status == "ok"
    assert out.result == {
        "taxable_value": "10000.00",
        "cgst": "900.00",
        "sgst": "900.00",
        "igst": "0.00",
        "total_tax": "1800.00",
        "total": "11800.00",
    }


def test_inter_state_is_igst_at_the_full_rate() -> None:
    out = gst_split(D("25000"), D("12"), "inter")
    assert (out.result["igst"], out.result["cgst"], out.result["sgst"]) == (
        "3000.00",
        "0.00",
        "0.00",
    )
    assert out.result["total"] == "28000.00"


def test_rounding_is_half_up_to_two_decimals() -> None:
    # 999.99 x 5 / 100 = 49.9995 -> 50.00
    assert gst_split(D("999.99"), D("5"), "intra").result["total_tax"] == "50.00"
    # 3333.33 x 28 / 100 = 933.3324 -> 933.33
    assert gst_split(D("3333.33"), D("28"), "inter").result["igst"] == "933.33"
    # 0.05 x 10 / 100 = 0.005 -> 0.01 (half up, not banker's rounding)
    assert gst_split(D("0.05"), D("10"), "inter").result["igst"] == "0.01"


def test_odd_paise_go_to_cgst() -> None:
    # 101 x 5 / 100 = 5.05; CGST = 2.525 -> 2.53, SGST = 5.05 - 2.53 = 2.52
    out = gst_split(D("101"), D("5"), "intra")
    assert (out.result["cgst"], out.result["sgst"], out.result["total_tax"]) == (
        "2.53",
        "2.52",
        "5.05",
    )


def test_inclusive_price_backs_out_the_tax() -> None:
    out = gst_split(D("1180"), D("18"), "intra", inclusive=True)
    assert out.result["taxable_value"] == "1000.00"
    assert (out.result["cgst"], out.result["sgst"], out.result["total"]) == (
        "90.00",
        "90.00",
        "1180.00",
    )
    inter = gst_split(D("5600"), D("12"), "inter", inclusive=True)
    assert (inter.result["taxable_value"], inter.result["igst"]) == ("5000.00", "600.00")


def test_inclusive_price_that_cannot_be_hit_exactly() -> None:
    # 100 x 100 / 118 = 84.7457...; 84.75 gives 100.01 and 84.74 gives 99.99 (both 0.01 off);
    # the tie goes to the value closer to 84.7457, i.e. 84.75.
    out = gst_split(D("100"), D("18"), "intra", inclusive=True)
    assert out.result["taxable_value"] == "84.75"
    assert (out.result["total_tax"], out.result["total"]) == ("15.26", "100.01")
    assert any("differs from the price" in step for step in out.steps)


def test_inputs_are_validated_and_strings_accepted() -> None:
    assert gst_split("1,180.00", "18", "intra", inclusive=True).result["taxable_value"] == "1000.00"
    with pytest.raises(ValueError):
        gst_split(D("100"), D("18"), "export")
    with pytest.raises(ValueError):
        gst_split(D("-1"), D("18"), "intra")
    with pytest.raises(ValueError):
        gst_split("abc", D("18"), "intra")
    with pytest.raises(ValueError):
        gst_split(D("100"), D("120"), "intra")


def test_citations_only_for_servable_index_rows() -> None:
    day = date(2025, 1, 1)
    snapshot = {
        "CGST/s9/ss1": [
            {
                "id": "CGST/s9/ss1@snapshot",
                "servable": "0",
                "effective_from": "2017-07-01",
                "effective_to": "9999-12-31",
            }
        ]
    }
    out = gst_split(D("100"), D("18"), "intra", on=day, provisions=snapshot)
    assert out.citations == []
    assert any("not a servable provision" in note for note in out.notes)
    servable = {
        "CGST/s9/ss1": [{**snapshot["CGST/s9/ss1"][0], "id": "CGST/s9/ss1@v1", "servable": "1"}]
    }
    assert gst_split(D("100"), D("18"), "intra", on=day, provisions=servable).citations == [
        "CGST/s9/ss1@v1"
    ]
    inter = gst_split(D("100"), D("18"), "inter", on=day, provisions=servable)
    assert inter.citations == [] and any("IGST Act" in note for note in inter.notes)


money = st.decimals(min_value=D("0"), max_value=D("10000000"), places=2, allow_nan=False)
rates = st.sampled_from([D(r) for r in ("0", "0.25", "1.5", "3", "5", "6", "12", "18", "28", "40")])


@settings(max_examples=300, deadline=None)
@given(value=money, rate=rates)
def test_property_cgst_plus_sgst_is_total_tax(value: Decimal, rate: Decimal) -> None:
    out = gst_split(value, rate, "intra").result
    assert D(out["cgst"]) + D(out["sgst"]) == D(out["total_tax"])
    assert abs(D(out["cgst"]) - D(out["sgst"])) <= D("0.01")
    assert D(out["taxable_value"]) + D(out["total_tax"]) == D(out["total"])


@settings(max_examples=300, deadline=None)
@given(price=money, rate=rates, supply=st.sampled_from(["intra", "inter"]))
def test_property_inclusive_then_exclusive_round_trips(
    price: Decimal, rate: Decimal, supply
) -> None:
    taxable = gst_split(price, rate, supply, inclusive=True).result["taxable_value"]
    total = D(gst_split(D(taxable), rate, supply).result["total"])
    assert abs(total - price) <= D("0.01")


@settings(max_examples=300, deadline=None)
@given(value=money, rate=rates, supply=st.sampled_from(["intra", "inter"]))
def test_property_exclusive_then_inclusive_recovers_the_value(value, rate, supply) -> None:
    total = gst_split(value, rate, supply).result["total"]
    assert gst_split(D(total), rate, supply, inclusive=True).result["taxable_value"] == str(value)


# --- rate_lookup -------------------------------------------------------------------------


def entry(
    s_no: int,
    tariff: str,
    description: str,
    rate: str,
    schedule: str = "I",
    start: date = date(2025, 9, 22),
    end: date | None = None,
    number: str = "0/0000",
) -> RateEntry:
    path = f"CTR/{number.replace('/', '-')}/sch{schedule}"
    return RateEntry(
        f"{path}@v1#{s_no}", number, schedule, s_no, tariff, description, rate, start, end
    )


ENTRIES = [
    entry(1, "1111", "Alpha widgets", "2.5"),
    entry(2, "1111 20", "Alpha widgets of beta metal", "9", schedule="II"),
    entry(3, "2222", "Gamma gadgets, other than delta gadgets", "9", schedule="II"),
    entry(4, "3333", "Gamma gizmos", "20", schedule="III"),
    entry(5, "Any Chapter", "Goods not elsewhere specified", "9", schedule="II"),
    entry(6, "4444", "Epsilon parts", "nil", schedule=""),
    entry(
        7,
        "1111",
        "Alpha widgets",
        "6",
        start=date(2017, 7, 1),
        end=date(2025, 9, 21),
        number="1/0000",
    ),
]
DAY = date(2025, 12, 1)
HORIZON = date(2026, 1, 31)


def look(*args, **kwargs):
    return rate_lookup(*args, entries=ENTRIES, horizon=HORIZON, **kwargs)


def test_unique_description_gives_rate_schedule_and_citation() -> None:
    out = look("gamma gizmos", on=DAY)
    assert out.status == "ok"
    assert {k: out.result[k] for k in ("rate", "combined_rate", "schedule", "s_no")} == {
        "rate": "20",
        "combined_rate": "40",
        "schedule": "III",
        "s_no": 4,
    }
    assert out.citations == ["CTR/0-0000/schIII@v1#4"]


def test_exact_description_wins_over_longer_ones() -> None:
    # "Alpha widgets" is a whole description (row 1) and part of row 2.
    out = look("alpha widgets", on=DAY)
    assert out.status == "ok" and out.result["s_no"] == 1


def test_ambiguous_description_returns_all_candidates_and_no_rate() -> None:
    out = look("gamma", on=DAY)
    assert out.status == "ambiguous" and out.result is None
    assert sorted(c["s_no"] for c in out.candidates) == [3, 4]


def test_hsn_prefers_the_most_specific_code_and_never_matches_any_chapter() -> None:
    assert look(hsn="1111 20 90", on=DAY).result["s_no"] == 2
    assert look(hsn="1111 10", on=DAY).result["s_no"] == 1
    # No row covers 11; the rows whose code starts with 11 are candidates.
    assert sorted(c["s_no"] for c in look(hsn="11", on=DAY).candidates) == [1, 2]
    assert look(hsn="9999", on=DAY).status == "not_found"


def test_only_rows_valid_on_the_date_are_searched() -> None:
    old = look("alpha widgets", on=date(2020, 1, 1))
    assert (old.result["rate"], old.result["notification"]) == ("6", "1/0000")


def test_exempt_rows_have_nil_rate_and_zero_combined() -> None:
    out = look("epsilon parts", on=DAY)
    assert (out.result["rate"], out.result["combined_rate"], out.result["schedule"]) == (
        "nil",
        "0",
        "exemption",
    )


def test_refused_after_the_rates_horizon_with_the_date_gate_message() -> None:
    out = look("gamma gizmos", on=date(2026, 2, 1))
    assert out.status == "refused" and out.result is None
    assert out.notes == [
        "The reference date 2026-02-01 is after 2026-01-31, the last date the corpus covers "
        "for rate notifications, so the law on that date cannot be confirmed."
    ]
    assert look("gamma gizmos", on=HORIZON).status == "ok"


def test_lookup_arguments_are_validated() -> None:
    with pytest.raises(ValueError):
        look(on=DAY)
    with pytest.raises(ValueError):
        look("  ", hsn="", on=DAY)
    with pytest.raises(ValueError):
        look(hsn="1", on=DAY)


def test_word_folding() -> None:
    assert words("Ferro-alloys and Nails") == {"ferro", "alloy", "nail"}


def test_real_index_row() -> None:
    # data/raw/ctr-09-2025_2025-09-17.pdf, Schedule II - 9%: "261. 7202 Ferro-alloys".
    out = rate_lookup("ferro-alloys", on=date(2025, 12, 1))
    assert (out.result["rate"], out.result["schedule"], out.result["s_no"]) == ("9", "II", 261)
    assert out.citations == ["CTR/9-2025/schII@v1#261"]


# --- description + HSN, exclusions, leading words (synthetic rows) -------------------------

MORE = [
    entry(11, "5555", "Delta devices", "2.5"),
    entry(12, "5555", "Parts of delta devices", "9", schedule="II"),
    entry(13, "6666", "Omega units (except heavy omega units of high capacity)", "2.5"),
    entry(14, "6666", "Heavy omega units of high capacity", "9", schedule="II"),
    entry(15, "7777", "Sigma frames, comprising a beam and a plate", "9", schedule="II"),
    entry(
        16,
        "7778",
        "Kappa items; other kappa items other than sigma frames of heading 7777",
        "9",
        schedule="II",
    ),
]


def look_more(*args, **kwargs):
    return rate_lookup(*args, entries=MORE, horizon=HORIZON, **kwargs)


def test_description_and_hsn_together() -> None:
    # The HSN alone gives two rows; the description chooses between them.
    assert look_more(hsn="5555", on=DAY).status == "ambiguous"
    both = look_more("parts of delta devices", hsn="5555", on=DAY)
    assert both.status == "ok" and both.result["s_no"] == 12
    # A single HSN row wins even if the description is worded differently.
    assert look_more("kappa goods", hsn="7778", on=DAY).result["s_no"] == 16
    # An HSN that matches nothing falls back to the description.
    assert look_more("sigma frames", hsn="9999", on=DAY).result["s_no"] == 15


def test_exclusions_do_not_match_the_excluded_row() -> None:
    query = "omega units other than heavy omega units of high capacity"
    out = look_more(query, on=DAY)
    assert out.status == "ok" and out.result["s_no"] == 13
    bracketed = look_more("omega units (excluding heavy omega units of high capacity)", on=DAY)
    assert bracketed.result["s_no"] == 13


def test_a_row_naming_the_goods_only_in_its_exclusion_is_not_a_match() -> None:
    # Row 16 mentions sigma frames only in "other than sigma frames"; row 15 is the match.
    out = look_more("sigma frames", on=DAY)
    assert out.status == "ok" and out.result["s_no"] == 15


def test_leading_words_choose_the_heading_row() -> None:
    # "delta devices" is in both rows; only row 11 begins with it (row 12 is "Parts of ...").
    out = look_more("delta devices", on=DAY)
    assert out.status == "ok" and out.result["s_no"] == 11


# --- real index rows (data/raw/ctr-1-2017_2017-06-28.pdf, ctr-09-2025_2025-09-17.pdf) -------


def test_tractors_2025_do_not_return_the_road_tractors_row() -> None:
    # 9/2025 Schedule I S. No. 442: "8701 Tractors (except road tractors for semi-trailers of
    # engine capacity more than 1800 cc)"; Schedule II S. No. 531 is the road-tractors row.
    query = (
        "tractors other than road tractors for semi trailers of engine capacity more than 1800 cc"
    )
    out = rate_lookup(query, on=date(2025, 10, 1))
    assert (out.result["rate"], out.result["schedule"], out.result["s_no"]) == ("2.5", "I", 442)
    old = rate_lookup(query, on=date(2017, 7, 1))
    assert (old.result["rate"], old.result["schedule"], old.result["s_no"]) == ("6", "II", 207)


def test_air_conditioning_machines() -> None:
    # 8415 "Air-conditioning machines, comprising a motor-driven fan ..."; the 8418 row names
    # air-conditioning machines only in its "other than" clause.
    new = rate_lookup("air-conditioning machines", on=date(2025, 10, 1))
    assert (new.result["rate"], new.result["schedule"], new.result["s_no"]) == ("9", "II", 404)
    old = rate_lookup("air-conditioning machines", on=date(2017, 7, 1))
    assert (old.result["rate"], old.result["schedule"], old.result["s_no"]) == ("14", "IV", 119)


def test_real_description_and_hsn_call() -> None:
    # 9/2025 Schedule I S. No. 290: "4202 22 30 Hand bags and shopping bags, of jute"; the
    # heading row 4202 (Schedule II) also covers 42022230 but is less specific.
    out = rate_lookup("Hand bags and shopping bags, of jute", hsn="42022230", on=date(2025, 10, 1))
    assert (out.result["rate"], out.result["schedule"], out.result["s_no"]) == ("2.5", "I", 290)
