"""Unit tests for the FalkorDB QA chain helpers (no database required)."""

import pytest

from langchain_falkordb.chains.graph_qa import FalkorDBQAChain, extract_cypher


class TestExtractCypher:
    def test_no_fences(self) -> None:
        assert extract_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"

    def test_plain_fences(self) -> None:
        assert extract_cypher("```\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"

    def test_cypher_language_tag(self) -> None:
        text = "```cypher\nMATCH (n) RETURN n\n```"
        assert extract_cypher(text) == "MATCH (n) RETURN n"

    def test_surrounding_prose(self) -> None:
        text = "Here you go:\n```cypher\nMATCH (n) RETURN n\n```\nEnjoy!"
        assert extract_cypher(text) == "MATCH (n) RETURN n"


def test_dangerous_requests_gate() -> None:
    """The chain must refuse to build without the explicit opt-in."""

    class DummyGraph:
        get_schema = ""

    with pytest.raises(ValueError, match="allow_dangerous_requests"):
        FalkorDBQAChain(
            graph=DummyGraph(),
            cypher_generation_chain=None,  # type: ignore[arg-type]
            qa_chain=None,  # type: ignore[arg-type]
        )
