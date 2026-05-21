import pytest

from agents.query_judge_deepseek import (
    parse_single_variant_response,
    VARIANT_SPECS,
)


def test_parse_single_line_response():
    assert parse_single_variant_response("Procedures for travel insurance claim adjudication") == (
        "Procedures for travel insurance claim adjudication"
    )


def test_parse_quoted_bullet_after_thinking():
    response = """
<think>
reasoning...
</think>

1. **Legal Process for Travel Insurance Claims**
   - "Procedures for adjudication of travel insurance claims under applicable laws."
"""
    assert "adjudication" in parse_single_variant_response(response)


def test_parse_labeled_quoted_line():
    response = (
        '1. **legal_terminology_rewrite**: '
        '"Which legal terms define claim submission processes?"'
    )
    assert "legal terms" in parse_single_variant_response(response)


def test_parse_requires_substantive_query():
    with pytest.raises(ValueError, match="No query line"):
        parse_single_variant_response("ok")


def test_four_variant_specs():
    assert len(VARIANT_SPECS) == 4
