"""Abstract graph store interface.

Vendored from ``langchain-community`` (MIT licensed) so that this package
only depends on ``langchain-core`` and the FalkorDB SDK.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from langchain_falkordb.graphs.graph_document import GraphDocument


@runtime_checkable
class GraphStore(Protocol):
    """Abstract class for graph operations."""

    @property
    def get_schema(self) -> str:
        """Return the schema of the Graph database."""
        ...

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Return the structured schema of the Graph database."""
        ...

    def query(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        """Query the graph."""
        ...

    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        ...

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input and use it to construct a graph."""
        ...
