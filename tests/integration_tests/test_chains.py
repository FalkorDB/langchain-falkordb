"""Integration tests for FalkorDBQAChain against a running FalkorDB.

The LLM is faked with canned responses; the graph queries are real.
"""

import os
from typing import Generator

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_falkordb.chains.graph_qa import FalkorDBQAChain
from langchain_falkordb.graphs import FalkorDBGraph

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", 6379))

CYPHER = (
    "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
    "WHERE m.title = 'Forrest Gump' RETURN a.name"
)


@pytest.fixture()
def graph() -> Generator[FalkorDBGraph, None, None]:
    wrapper = FalkorDBGraph("qa_chain_test", host=host, port=port)
    try:
        wrapper.query("MATCH (n) DELETE n")
        wrapper.query(
            "CREATE (:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->"
            "(:Movie {title: 'Forrest Gump'})"
        )
        wrapper.refresh_schema()
        yield wrapper
    finally:
        try:
            wrapper._graph.delete()
        except Exception:
            pass


def test_qa_chain_answers_from_graph(graph: FalkorDBGraph) -> None:
    llm = FakeListChatModel(responses=[CYPHER, "Tom Hanks acted in Forrest Gump."])
    chain = FalkorDBQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
    output = chain.invoke({"query": "Who acted in Forrest Gump?"})
    assert output["result"] == "Tom Hanks acted in Forrest Gump."


def test_qa_chain_accepts_plain_string_input(graph: FalkorDBGraph) -> None:
    llm = FakeListChatModel(responses=[CYPHER, "Tom Hanks."])
    chain = FalkorDBQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
    output = chain.invoke("Who acted in Forrest Gump?")
    assert output["result"] == "Tom Hanks."


def test_qa_chain_extracts_fenced_cypher(graph: FalkorDBGraph) -> None:
    llm = FakeListChatModel(responses=[f"```cypher\n{CYPHER}\n```", "Tom Hanks."])
    chain = FalkorDBQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
    output = chain.invoke({"query": "Who acted in Forrest Gump?"})
    assert output["result"] == "Tom Hanks."


def test_qa_chain_return_direct(graph: FalkorDBGraph) -> None:
    llm = FakeListChatModel(responses=[CYPHER])
    chain = FalkorDBQAChain.from_llm(
        llm, graph=graph, allow_dangerous_requests=True, return_direct=True
    )
    output = chain.invoke({"query": "Who acted in Forrest Gump?"})
    assert output["result"] == [["Tom Hanks"]]


def test_qa_chain_intermediate_steps(graph: FalkorDBGraph) -> None:
    llm = FakeListChatModel(responses=[CYPHER, "Tom Hanks."])
    chain = FalkorDBQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
    )
    output = chain.invoke({"query": "Who acted in Forrest Gump?"})
    assert output["intermediate_steps"] == [
        {"query": CYPHER},
        {"context": [["Tom Hanks"]]},
    ]


def test_qa_chain_top_k(graph: FalkorDBGraph) -> None:
    graph.query("CREATE (:Actor {name: 'Robin Wright'})-[:ACTED_IN]->(m:Movie)")
    llm = FakeListChatModel(responses=["MATCH (a:Actor) RETURN a.name ORDER BY a.name"])
    chain = FalkorDBQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        return_direct=True,
        top_k=1,
    )
    output = chain.invoke({"query": "List actors"})
    assert len(output["result"]) == 1
