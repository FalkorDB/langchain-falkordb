"""Graph document models.

Vendored from ``langchain-community`` (MIT licensed) so that this package
only depends on ``langchain-core`` and the FalkorDB SDK.
"""

from __future__ import annotations

from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable
from pydantic import Field


class Node(Serializable):
    """Represents a node in a graph with associated properties.

    Attributes:
        id: A unique identifier for the node.
        type: The type or label of the node, default is "Node".
        properties: Additional properties and metadata associated with the
            node.
    """

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source: The source node of the relationship.
        target: The target node of the relationship.
        type: The type of the relationship.
        properties: Additional properties associated with the relationship.
    """

    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)


class GraphDocument(Serializable):
    """Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes: A list of nodes in the graph.
        relationships: A list of relationships in the graph.
        source: The document from which the graph information is derived.
    """

    nodes: List[Node]
    relationships: List[Relationship]
    source: Optional[Document] = None
