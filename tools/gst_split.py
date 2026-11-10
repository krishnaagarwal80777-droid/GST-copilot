"""GST amounts from a taxable value (or a tax-inclusive price) and a combined GST rate.

    gst_split(Decimal("10000"), Decimal("18"), "intra")
    -> result {taxable_value: 10000.00, cgst: 900.00, sgst: 900.00, igst: 0.00,
               total_tax: 1800.00, total: 11800.00}

``rate`` is the combined GST rate in per cent: intra-State supplies bear CGST and SGST at half
the rate each; inter-State supplies bear IGST at the full rate.

Rounding assumption (no paise-level rounding rule is taken from the law here):
- All arithmetic is in Decimal; amounts are rounded to 2 decimal places with ROUND_HALF_UP.
- The tax is computed once on the combined rate: total_tax = round(value x rate / 100).
- Intra-State: CGST = round(total_tax / 2) and SGST = total_tax - CGST, so CGST + SGST always
  equals total_tax; when total_tax has an odd number of paise, CGST is 0.01 higher than SGST.
- Tax-inclusive price (``inclusive=True``): the taxable value is the 2-decimal amount whose
  total (value + tax, computed as above) is closest to the price; ties go to the value closest
  to price x 100 / (100 + rate). The total can differ from the price by at most 0.01.

Citations: the governing provisions are cited only when they are servable rows of the index
(``provisions``: index path -> rows). Section 9(1) of the CGST Act levies central tax on
intra-State supplies; the IGST Act and the State GST Acts are not in the index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal

from tools.result import ToolResult

CENT = Decimal("0.01")
HUNDRED = Decimal("100")
Supply = Literal["intra", "inter"]
# Index paths of the provisions a split rests on, by supply type.
GOVERNING: dict[str, tuple[str, ...]] = {"intra": ("CGST/s9/ss1",), "inter": ()}
NOT_INDEXED: dict[str, str] = {
    "intra": "State tax (SGST) is levied under the State GST Acts, which are not in the index.",
    "inter": "Integrated tax (IGST) is levied under the IGST Act, which is not in the index.",
}


def to_decimal(value: Any, name: str) -> Decimal:
    """Decimal from a Decimal, int or numeric string ("1,180.50" allowed); floats go through
    their string form."""
    if isinstance(value, Decimal):
        number = value
    else:
        try:
            number = Decimal(str(value).replace(",", "").replace("₹", "").strip())
        except InvalidOperation:
            raise ValueError(f"{name} is not a number: {value!r}") from None
    if not number.is_finite():
        raise ValueError(f"{name} is not a finite number: {value!r}")
    return number


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _amounts(value: Decimal, rate: Decimal, supply: str) -> dict[str, Decimal]:
    total_tax = money(value * rate / HUNDRED)
    if supply == "intra":
        cgst = money(total_tax / 2)
        sgst, igst = total_tax - cgst, Decimal("0.00")
    else:
        cgst = sgst = Decimal("0.00")
        igst = total_tax
    return {"cgst": cgst, "sgst": sgst, "igst": igst, "total_tax": total_tax}


def _back_out(price: Decimal, rate: Decimal, supply: str) -> tuple[Decimal, Decimal]:
    """(taxable value, exact price x 100 / (100 + rate)) for a tax-inclusive price."""
    exact = price * HUNDRED / (HUNDRED + rate)
    start = money(exact)
    candidates = [start + CENT * step for step in range(-3, 4) if start + CENT * step >= 0]

    def gap(value: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        total = value + _amounts(value, rate, supply)["total_tax"]
        return abs(total - price), abs(value - exact), value

    return min(candidates, key=gap), exact


def _citations(
    supply: str, on: date | None, provisions: Mapping[str, Sequence[dict]] | None
) -> tuple[list[str], list[str]]:
    notes = [NOT_INDEXED[supply]]
    if provisions is None or on is None:
        notes.append("Governing provisions were not looked up (no index or date given).")
        return [], notes
    day = on.isoformat()
    cited: list[str] = []
    for path in GOVERNING[supply]:
        rows = [
            row
            for row in provisions.get(path, [])
            if str(row.get("servable", "1")) == "1"
            and row["effective_from"] <= day <= row["effective_to"]
        ]
        if rows:
            cited.append(rows[0]["id"])
        else:
            notes.append(
                f"{path} is not a servable provision of the index on {day}, so it is not cited."
            )
    return cited, notes


def gst_split(
    taxable_value: Decimal | str | int,
    rate: Decimal | str | int,
    supply: Supply,
    *,
    inclusive: bool = False,
    on: date | None = None,
    provisions: Mapping[str, Sequence[dict]] | None = None,
) -> ToolResult:
    """CGST/SGST (intra-State) or IGST (inter-State) on a taxable value at a combined GST rate;
    with ``inclusive`` the first argument is a tax-inclusive price and the tax is backed out."""
    amount = to_decimal(taxable_value, "taxable_value")
    rate = to_decimal(rate, "rate")
    if supply not in ("intra", "inter"):
        raise ValueError(f"supply must be 'intra' or 'inter', not {supply!r}")
    if amount < 0:
        raise ValueError("taxable_value must not be negative")
    if not Decimal("0") <= rate <= HUNDRED:
        raise ValueError("rate must be a percentage between 0 and 100")

    steps: list[str] = []
    if inclusive:
        value, exact = _back_out(amount, rate, supply)
        steps.append(
            f"Tax-inclusive price {money(amount)} at {rate} per cent: price x 100 / "
            f"(100 + {rate}) = {exact.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}; "
            f"taxable value {value} (the 2-decimal value whose total is closest to the price)."
        )
    else:
        value = money(amount)
    taxes = _amounts(value, rate, supply)
    total = value + taxes["total_tax"]
    steps.append(
        f"Total tax = {value} x {rate} / 100 = {taxes['total_tax']} "
        "(rounded to 2 decimals, half up)."
    )
    if supply == "intra":
        steps.append(
            f"Intra-State supply: CGST and SGST at {rate / 2} per cent each; CGST = "
            f"{taxes['total_tax']} / 2 rounded = {taxes['cgst']}, SGST = {taxes['total_tax']} "
            f"- {taxes['cgst']} = {taxes['sgst']}."
        )
    else:
        steps.append(f"Inter-State supply: IGST at {rate} per cent = {taxes['igst']}.")
    steps.append(f"Total = {value} + {taxes['total_tax']} = {total}.")
    if inclusive and total != money(amount):
        steps.append(f"The total differs from the price {money(amount)} by rounding.")
    citations, notes = _citations(supply, on, provisions)
    notes.append(
        "Amounts use Decimal arithmetic rounded to 2 decimals with ROUND_HALF_UP; the law "
        "itself is not the source of this paise-level rounding."
    )
    result = {"taxable_value": value, **taxes, "total": total}
    return ToolResult(
        tool="gst_split",
        status="ok",
        result={key: str(number) for key, number in result.items()},
        steps=steps,
        citations=citations,
        notes=notes,
    )
