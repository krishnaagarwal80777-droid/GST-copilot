from ingest.segment import parse_act_text


def test_stack_parser_builds_canonical_paths_from_synthetic_text() -> None:
    fixture = """\
Section TEST. Synthetic parser fixture
(1) synthetic subsection text
(a) synthetic clause text
(i) synthetic sub-clause text
Provided that synthetic proviso text
Explanation. synthetic explanation text
(ii) another synthetic sub-clause
(b) sibling synthetic clause
(i) sibling synthetic sub-clause
Section SAMPLE. Another synthetic parser fixture
(1) another synthetic subsection
"""

    nodes = parse_act_text(fixture)

    assert [node.path for node in nodes] == [
        "CGST/sTEST",
        "CGST/sTEST/ss1",
        "CGST/sTEST/ss1/cl(a)",
        "CGST/sTEST/ss1/cl(a)/scl(i)",
        "CGST/sTEST/ss1/cl(a)/scl(i)/proviso1",
        "CGST/sTEST/ss1/cl(a)/scl(i)/explanation1",
        "CGST/sTEST/ss1/cl(a)/scl(ii)",
        "CGST/sTEST/ss1/cl(b)",
        "CGST/sTEST/ss1/cl(b)/scl(i)",
        "CGST/sSAMPLE",
        "CGST/sSAMPLE/ss1",
    ]
    assert nodes[0].heading == "Synthetic parser fixture"
    assert nodes[4].parent_path == "CGST/sTEST/ss1/cl(a)/scl(i)"
    assert nodes[4].text == "synthetic proviso text"
    assert nodes[-1].text == "another synthetic subsection"


def test_stack_parser_handles_section_16_and_17_style_headers() -> None:
    fixture = """\
Section 16. Parser fixture for section 16
(1) first subsection
(a) first clause
(i) first sub-clause
Provided that first proviso text
Explanation. first explanation text
Section 17. Parser fixture for section 17
(1) second subsection
(b) second clause
"""

    nodes = parse_act_text(fixture)

    assert [node.path for node in nodes[:8]] == [
        "CGST/s16",
        "CGST/s16/ss1",
        "CGST/s16/ss1/cl(a)",
        "CGST/s16/ss1/cl(a)/scl(i)",
        "CGST/s16/ss1/cl(a)/scl(i)/proviso1",
        "CGST/s16/ss1/cl(a)/scl(i)/explanation1",
        "CGST/s17",
        "CGST/s17/ss1",
    ]
    assert nodes[0].heading == "Parser fixture for section 16"
    assert nodes[6].heading == "Parser fixture for section 17"
    assert nodes[4].text == "first proviso text"
