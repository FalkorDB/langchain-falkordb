"""Unit tests for the graph document models (no database required)."""

from langchain_core.documents import Document

from langchain_falkordb.graphs import GraphDocument, Node, Relationship


def test_node_defaults() -> None:
    node = Node(id="n1")
    assert node.id == "n1"
    assert node.type == "Node"
    assert node.properties == {}


def test_graph_document_model() -> None:
    alice = Node(id="alice", type="Person")
    bob = Node(id="bob", type="Person")
    relationship = Relationship(
        source=alice, target=bob, type="KNOWS", properties={"since": 2020}
    )
    document = GraphDocument(
        nodes=[alice, bob],
        relationships=[relationship],
        source=Document(page_content="Alice knows Bob."),
    )
    assert document.nodes[0].id == "alice"
    assert document.relationships[0].properties["since"] == 2020
    assert document.source is not None
    assert document.source.page_content == "Alice knows Bob."
