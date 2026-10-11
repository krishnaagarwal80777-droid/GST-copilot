from __future__ import annotations

import re
from collections.abc import Sequence

from gst_copilot.models import ProvisionRecord

SECTION_RE = re.compile(r"(?:section|sec|s)\s*(\d+[A-Z]?)\s*(?:\((\d+[A-Z]?)\))?", re.IGNORECASE)
RULE_RE = re.compile(r"(?:rule)\s*(\d+[A-Z]?)", re.IGNORECASE)


def _section_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for match in SECTION_RE.finditer(query):
        section_number = match.group(1)
        subsection_number = match.group(2)
        if subsection_number:
            tokens.add(f"CGST/s{section_number}/ss{subsection_number}")
            tokens.add(f"CGST/s{section_number}")
        else:
            tokens.add(f"CGST/s{section_number}")
    return tokens


def _rule_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for match in RULE_RE.finditer(query):
        tokens.add(f"CGST/r{match.group(1)}")
    return tokens


def exact_match(query: str, records: Sequence[ProvisionRecord]) -> list[ProvisionRecord]:
    """Return records that match explicit legal references such as Section 17(5) or Rule 42."""
    section_tokens = _section_tokens(query)
    rule_tokens = _rule_tokens(query)

    if not section_tokens and not rule_tokens:
        return []

    query_keywords = {word for word in re.findall(r"[A-Za-z]+", query.lower()) if len(word) > 2}
    matches: list[tuple[int, ProvisionRecord]] = []
    for record in records:
        record_text = " ".join(filter(None, [record.path, record.heading or "", record.text]))
        score = 0
        lower_text = record_text.lower()
        for token in section_tokens:
            token_l = token.lower()
            if record.path.lower() == token_l:
                score += 100
            elif token_l in record.path.lower():
                score += 60
                if "/" in record.path:
                    score += 25
            if token_l in lower_text:
                score += 15
        for token in rule_tokens:
            token_l = token.lower()
            if record.path.lower() == token_l:
                score += 100
            elif token_l in record.path.lower():
                score += 60
                if "/" in record.path:
                    score += 25
            if token_l in lower_text:
                score += 15

        keyword_hits = sum(1 for keyword in query_keywords if keyword in lower_text)
        score += keyword_hits * 5
        if "/" in record.path:
            score += 10
        if "/ss" in record.path or "/cl(" in record.path or "/proviso" in record.path:
            score += 25

        if score:
            matches.append((score, record))

    matches.sort(key=lambda item: (-item[0], -item[1].path.count("/"), item[1].path))
    return [record for _, record in matches]


# --- Citation normalisation for the exact-match path -------------------------------------
_SUBDIVISIONS = r"(?P<subs>(?:\s*\(\s*[0-9A-Za-z]{1,4}\s*\))*)"
_RULE_CITE = re.compile(r"\brules?\s+(?P<num>\d{1,3}[A-Z]{0,2})\b" + _SUBDIVISIONS, re.IGNORECASE)
_SECTION_CITE = re.compile(
    r"(?:\b(?:section|sec\.?|s\.)\s*)?(?<![\w/-])(?P<num>\d{1,3}[A-Z]?)" + _SUBDIVISIONS,
    re.IGNORECASE,
)
_SECTION_WORD = re.compile(r"(?:\b(?:section|sec\.?|s\.)\s*)$", re.IGNORECASE)
_NOTIFICATION_CITE = re.compile(
    r"\bNotification\s+(?:No\.?\s*)?(?P<num>\d{1,3})/(?P<year>\d{4})\s*[-\u2013]?\s*"
    r"(?:CT|Central\s+Tax)\s*\(\s*Rate\s*\)",
    re.IGNORECASE,
)


def _subdivision_path(prefix: str, subs: str) -> str:
    """Append (4) as /ss4 and (a)/(aa) as /cl(a); stop at anything else."""
    path = prefix
    for index, token in enumerate(re.findall(r"\(\s*([0-9A-Za-z]{1,4})\s*\)", subs)):
        if index == 0 and re.fullmatch(r"\d+[A-Z]?", token):
            path += f"/ss{token}"
        elif index > 0 and re.fullmatch(r"[a-z]{1,2}", token):
            path += f"/cl({token})"
        else:
            break
    return path


def parse_citations(query: str) -> list[str]:
    """Canonical paths cited in a query, in order of appearance.

    "17(5)", "s.16(4)", "section 17(5)(a)" -> CGST/s17/ss5[/cl(a)]; "rule 36(4)" ->
    CGST/r36/ss4; "Notification 9/2025-CT(Rate)" -> CTR/9-2025. A bare number counts as a
    section only when it carries a sub-section, e.g. "16(4)", so years and amounts are ignored.
    """
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for match in _NOTIFICATION_CITE.finditer(query):
        found.append((match.start(), f"CTR/{int(match.group('num'))}-{match.group('year')}"))
        taken.append(match.span())
    for match in _RULE_CITE.finditer(query):
        if any(start <= match.start() < end for start, end in taken):
            continue
        found.append(
            (match.start(), _subdivision_path(f"CGST/r{match.group('num')}", match.group("subs")))
        )
        taken.append(match.span())
    for match in _SECTION_CITE.finditer(query):
        if any(start <= match.start("num") < end for start, end in taken):
            continue
        has_word = _SECTION_WORD.search(query[: match.start("num")]) is not None
        if not match.group("subs").strip() and not has_word:
            continue
        found.append(
            (match.start(), _subdivision_path(f"CGST/s{match.group('num')}", match.group("subs")))
        )
    return list(dict.fromkeys(path for _, path in sorted(found)))


def exact_paths(cited: str, indexed_paths: Sequence[str]) -> list[str]:
    """Indexed paths for one citation: the path itself, then its descendants (shallowest first).

    A citation deeper than anything indexed (e.g. "rule 89(1)" when only rule 89 is indexed)
    falls back to its nearest indexed ancestor.
    """
    probe = cited
    while probe.count("/") >= 1:
        hits = [path for path in indexed_paths if path == probe or path.startswith(probe + "/")]
        if hits:
            return sorted(set(hits), key=lambda path: (path != probe, path.count("/"), path))
        probe = probe.rsplit("/", 1)[0]
    return []
