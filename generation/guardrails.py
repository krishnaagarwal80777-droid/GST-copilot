"""Input guardrails: question length, prompt injection, out of scope. No LLM call is made for a
refused question; each guardrail is a flag and has a fixed refusal message.

Order: length, injection, keyword scope (before retrieval), then retrieval-confidence scope.

Scope from keywords is three-way:
- out: a foreign tax system is named (GST also exists elsewhere; a country named next to an
  import or export word does not count); or another tax (Indian income tax, which v1 does
  not cover) is named without GST; or an off-topic keyword is present and no GST keyword is.
- in: a GST name (GST, CGST, IGST, ...) or a GST topic word (input tax credit, place of
  supply, e-way bill, ...) is present.
- unknown: neither. The question is retrieved and refused as out of scope when the best
  cross-encoder score is below ``scope_threshold``. An in-scope question with a low score is
  not refused here; the confidence gate abstains on it as usual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from generation.config import TUNED_ABSTAIN_THRESHOLD

Guardrail = Literal["max_length", "injection", "out_of_scope"]
Scope = Literal["in", "out", "unknown"]

REFUSALS: dict[str, str] = {
    "max_length": "The question is too long. Please ask one GST question in under {limit} "
    "characters.",
    "injection": "This request asks the assistant to change its instructions or role, so it "
    "was not processed. Please ask a question about Indian GST law.",
    "out_of_scope": "This assistant answers questions about Indian GST law only (the CGST and "
    "IGST Acts, the CGST Rules and central tax rate notifications).",
}


@dataclass(frozen=True)
class GuardrailConfig:
    max_length: bool = True
    max_question_chars: int = 1000
    injection: bool = True
    out_of_scope: bool = True
    # Same cross-encoder scale as the confidence gate; the dev-tuned gate threshold is used
    # (not tuned on the guardrail set).
    scope_threshold: float = TUNED_ABSTAIN_THRESHOLD


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


INJECTION = _any(
    r"\b(?:ignore|disregard|forget|skip|bypass)\b.{0,30}\b(?:previous|prior|above|earlier|"
    r"preceding|all|your|the|these|those)\b.{0,20}\b(?:instructions?|prompts?|rules|"
    r"guidelines|directions|messages|context)\b",
    r"\bsystem\s*prompt\b",
    r"\b(?:reveal|show|print|repeat|display|output|leak|dump|tell me|what (?:is|are)|what's)\b"
    r".{0,25}\b(?:your|the)\s+(?:hidden\s+|initial\s+|original\s+|internal\s+)?"
    r"(?:instructions|prompt|rules|configuration)\b",
    r"\byou are (?:now|no longer)\b|\byou're now\b|\bfrom now on,? you\b",
    r"\bpretend\b|\brole[- ]?play\b|\bstay in character\b|\bimagine you are\b",
    r"\bact as (?:if|though|my|an? (?:ai|assistant|chatbot|different|unrestricted|unfiltered|"
    r"evil|new))\b",
    r"\byour (?:new )?(?:role|persona|identity) is\b",
    r"\b(?:developer|god|admin|debug|jailbreak|dan) mode\b|\bjailbreak\b|\bDAN\b",
    r"\bnew instructions?\s*:",
    r"\boverride (?:your|the|all) (?:rules|instructions|guidelines|safety|restrictions)\b",
    r"<\|?\s*(?:im_start|system|endoftext)\s*\|?>|\[/?(?:INST|SYS)\]",
    r"^\s*(?:system|assistant)\s*:",
)

# Another country's tax system. A country alone is not enough: imports and exports are GST
# questions, so a country counts only when no cross-border trade word is present.
FOREIGN_AUTHORITY = _any(
    r"\b(?:irs|hmrc|iras|401\s*\(?k\)?|w-?2|1099|social security number)\b",
)
FOREIGN_PLACE = _any(
    r"\b(?:australia|australian|canada|canadian|new zealand|singapore|malaysia|malaysian|"
    r"united states|usa|america|american|united kingdom|uk|britain|british)\b",
)
CROSS_BORDER = _any(
    r"\bexport|\bimport|\boverseas\b|\boutside india\b|\bforeign (?:client|customer|supplier)\b",
    r"\bzero[- ]rated\b|\boidar\b|\blut\b",
)
OTHER_TAX = _any(
    r"\bincome[- ]?tax\b|\bitr\b|\bsection 80[a-z]+\b|\b80c\b|\bcapital gains?\b|\bhra\b",
    r"\bsalary\b|\bsalaried\b",
    r"\btds on (?:salary|rent|interest)\b|\bprofessional tax\b|\bproperty tax\b|"
    r"\bstamp duty\b|\bcustoms duty\b|\bwealth tax\b",
)
GST_NAME = _any(
    r"\b(?:gst|cgst|igst|sgst|utgst|gstin|gstr-?\d*[a-z]?|gstn|gstat)\b",
    r"\bgoods and services tax\b|\b(?:central|integrated|state|union territory) tax\b",
)
GST_TOPIC = _any(
    r"\binput tax credit\b|\bitc\b|\bplace of supply\b|\btime of supply\b|\bvalue of supply\b",
    r"\be-?way bill\b|\be-?invoice\b|\btax invoice\b|\b(?:credit|debit) note\b",
    r"\breverse charge\b|\bcomposition (?:scheme|levy|dealer|taxpayer)\b",
    r"\baggregate turnover\b|\binput service distributor\b|\bjob work(?:er)?\b",
    r"\b(?:works contract|e-commerce operator|zero[- ]rated|nil[- ]rated|inter-?state|"
    r"intra-?state)\b",
    r"\bhsn\b|\bsac\b|\bcompensation cess\b|\bcbic\b",
    r"\b(?:rule|section|notification|circular) (?:no\.? ?)?\d+",
    r"\b(?:taxable|exempt) supply\b|\bsupply of (?:goods|services)\b|\b(?:tax|rate) on\b",
    r"\bregistration\b.{0,30}\b(?:supplier|business|turnover|threshold)\b",
)
OFF_TOPIC = _any(
    r"\b(?:recipe|cook|bake|weather|forecast|cricket|football|ipl|movie|film|song|lyrics|"
    r"poem|joke|horoscope|astrology|workout|diet|calories|vacation|itinerary|dating)\b",
    r"\b(?:stock price|share price|bitcoin|crypto(?:currency)?|nifty|sensex)\b",
    r"\b(?:python|javascript|java|sql|html|css|code|program|algorithm)\b",
    r"\b(?:capital of|translate|president|prime minister|election|history of)\b",
)


def is_injection(question: str) -> bool:
    return INJECTION.search(question) is not None


def keyword_scope(question: str) -> Scope:
    if FOREIGN_AUTHORITY.search(question):
        return "out"
    if FOREIGN_PLACE.search(question) and not CROSS_BORDER.search(question):
        return "out"
    gst_name = GST_NAME.search(question) is not None
    if OTHER_TAX.search(question) and not gst_name:
        return "out"
    if gst_name or GST_TOPIC.search(question):
        return "in"
    if OFF_TOPIC.search(question):
        return "out"
    return "unknown"


def refusal(guardrail: Guardrail, config: GuardrailConfig) -> str:
    return REFUSALS[guardrail].format(limit=config.max_question_chars)


def check_before_retrieval(question: str, config: GuardrailConfig) -> Guardrail | None:
    if config.max_length and len(question) > config.max_question_chars:
        return "max_length"
    if config.injection and is_injection(question):
        return "injection"
    if config.out_of_scope and keyword_scope(question) == "out":
        return "out_of_scope"
    return None


def check_after_retrieval(
    question: str, confidence: float, config: GuardrailConfig
) -> Guardrail | None:
    if (
        config.out_of_scope
        and keyword_scope(question) == "unknown"
        and confidence < config.scope_threshold
    ):
        return "out_of_scope"
    return None
