"""Integration tests for the FalkorDBGraph wrapper.

These tests require a running FalkorDB instance on ``localhost:6379``
(override with ``FALKORDB_HOST`` / ``FALKORDB_PORT``).
"""

import os
from typing import Generator

import pytest
from langchain_core.documents import Document

from langchain_falkordb.graphs import (
    FalkorDBGraph,
    GraphDocument,
    Node,
    Relationship,
)

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", 6379))


@pytest.fixture()
def graph() -> Generator[FalkorDBGraph, None, None]:
    wrapper = FalkorDBGraph("graph_wrapper_test", host=host, port=port)
    try:
        wrapper.query("MATCH (n) DELETE n")
        yield wrapper
    finally:
        try:
            wrapper._graph.delete()
        except Exception:
            # Best-effort cleanup: the graph may never have been
            # created server-side, which is fine.
            pass


def test_query(graph: FalkorDBGraph) -> None:
    graph.query("CREATE (:Person {name: 'Alice', age: 30})")
    result = graph.query("MATCH (p:Person) RETURN p.name, p.age")
    assert result == [["Alice", 30]]


def test_driver_kwargs_passthrough() -> None:
    """Extra kwargs (e.g. timeouts) are forwarded to the FalkorDB client."""
    wrapper = FalkorDBGraph(
        "graph_kwargs_test",
        host=host,
        port=port,
        socket_connect_timeout=5,
        socket_timeout=30,
    )
    try:
        assert wrapper.query("RETURN 1") == [[1]]
    finally:
        try:
            wrapper._graph.delete()
        except Exception:
            # Best-effort cleanup: the graph may never have been
            # created server-side, which is fine.
            pass


def test_query_params(graph: FalkorDBGraph) -> None:
    graph.query("CREATE (:Person {name: $name})", {"name": "Bob"})
    result = graph.query(
        "MATCH (p:Person) WHERE p.name = $name RETURN count(p)", {"name": "Bob"}
    )
    assert result == [[1]]


def test_invalid_query_raises(graph: FalkorDBGraph) -> None:
    with pytest.raises(ValueError):
        graph.query("THIS IS NOT CYPHER")


def test_refresh_schema(graph: FalkorDBGraph) -> None:
    graph.query(
        "CREATE (:Person {name: 'Alice'})-[:KNOWS {since: 2020}]->"
        "(:Person {name: 'Bob'})"
    )
    graph.refresh_schema()
    structured = graph.get_structured_schema
    assert "Person" in structured["node_props"]
    assert "name" in structured["node_props"]["Person"]
    assert "KNOWS" in structured["rel_props"]
    assert {"start": "Person", "type": "KNOWS", "end": "Person"} in structured[
        "relationships"
    ]
    assert "Person" in graph.get_schema


def test_add_graph_documents(graph: FalkorDBGraph) -> None:
    alice = Node(id="Conan O'Brien", type="Person", properties={"role": "host"})
    show = Node(id="Late Night", type="Show")
    document = GraphDocument(
        nodes=[alice, show],
        relationships=[
            # Type with a space must be normalized to HOSTED_SHOW
            Relationship(
                source=alice, target=show, type="hosted show", properties={"y": 1993}
            )
        ],
        source=Document(page_content="Conan O'Brien hosted Late Night."),
    )
    graph.add_graph_documents([document])

    # Node ids containing quotes must round-trip (no string interpolation).
    people = graph.query("MATCH (p:Person) RETURN p.id, p.role")
    assert people == [["Conan O'Brien", "host"]]
    rels = graph.query("MATCH (:Person)-[r:HOSTED_SHOW]->(s:Show) RETURN r.y, s.id")
    assert rels == [[1993, "Late Night"]]


def test_add_graph_documents_include_source(graph: FalkorDBGraph) -> None:
    node = Node(id="FalkorDB", type="Product")
    document = GraphDocument(
        nodes=[node],
        relationships=[],
        source=Document(page_content="FalkorDB is fast.", metadata={"lang": "en"}),
    )
    graph.add_graph_documents([document], include_source=True)

    mentions = graph.query(
        "MATCH (d:Document)-[:MENTIONS]->(p:Product) RETURN d.text, p.id"
    )
    assert mentions == [["FalkorDB is fast.", "FalkorDB"]]


def test_add_graph_documents_rejects_backtick_labels(graph: FalkorDBGraph) -> None:
    bad = Node(id="x", type="Bad`) DELETE (n")
    document = GraphDocument(nodes=[bad], relationships=[], source=None)
    with pytest.raises(ValueError, match="backtick"):
        graph.add_graph_documents([document])
