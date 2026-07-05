"""FalkorDB graph wrapper for LangChain."""

import os
from hashlib import md5
from typing import Any, Dict, List, Optional

from langchain_falkordb.graphs.graph_document import GraphDocument

NODE_PROPERTIES_QUERY = """
MATCH (n)
WITH keys(n) as keys, labels(n) AS labels
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, labels
UNWIND labels AS label
UNWIND keys AS key
WITH label, collect(DISTINCT key) AS keys
RETURN {label:label, keys:keys} AS output
"""

REL_PROPERTIES_QUERY = """
MATCH ()-[r]->()
WITH keys(r) as keys, [type(r)] AS types
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, types
UNWIND types AS type
UNWIND keys AS key WITH type,
collect(DISTINCT key) AS keys
RETURN {types:type, keys:keys} AS output
"""

REL_QUERY = """
MATCH (n)-[r]->(m)
UNWIND labels(n) as src_label
UNWIND labels(m) as dst_label
UNWIND [type(r)] as rel_type
RETURN DISTINCT {start: src_label, type: rel_type, end: dst_label} AS output
"""


def _validated_label(label: str) -> str:
    """Validate a label/relationship type used inside backtick quotes."""
    if not label:
        raise ValueError("Labels and relationship types must not be empty")
    if "`" in label:
        raise ValueError(
            f"Label or relationship type {label!r} must not contain backtick characters"
        )
    return label


class FalkorDBGraph:
    """FalkorDB wrapper for graph operations.

    Args:
        database: The name of the graph to use.
        host: FalkorDB host. Defaults to ``"localhost"``.
        port: FalkorDB port. Defaults to ``6379``.
        username: Optionally provide your username. Falls back to the
            ``FALKORDB_USERNAME`` environment variable.
        password: Optionally provide your password. Falls back to the
            ``FALKORDB_PASSWORD`` environment variable.
        ssl: Whether the connection should use SSL/TLS encryption.
            Defaults to ``False``.
        kwargs: Additional keyword arguments forwarded to the
            ``falkordb.FalkorDB`` client (and ultimately redis-py), e.g.
            ``socket_timeout`` or ``socket_connect_timeout``.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the
        calling code may attempt commands that would result in deletion,
        mutation of data if appropriately prompted or reading sensitive data
        if such data is present in the database.
        The best way to guard against such negative outcomes is to (as
        appropriate) limit the permissions granted to the credentials used
        with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        database: str,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new FalkorDB graph wrapper instance."""
        try:
            from falkordb import FalkorDB
        except ImportError as e:
            raise ImportError(
                "Could not import falkordb python package. "
                "Please install it with `pip install falkordb`."
            ) from e

        try:
            self._driver = FalkorDB(
                host=host,
                port=port,
                username=username or os.environ.get("FALKORDB_USERNAME"),
                password=password or os.environ.get("FALKORDB_PASSWORD"),
                ssl=ssl,
                **kwargs,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to FalkorDB: {e}") from e
        self._graph = self._driver.select_graph(database)

        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}") from e

    @property
    def get_schema(self) -> str:
        """Return the schema of the FalkorDB database."""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Return the structured schema of the graph."""
        return self.structured_schema

    def refresh_schema(self) -> None:
        """Refresh the schema of the FalkorDB database."""
        node_properties: List[Any] = self.query(NODE_PROPERTIES_QUERY)
        rel_properties: List[Any] = self.query(REL_PROPERTIES_QUERY)
        relationships: List[Any] = self.query(REL_QUERY)

        self.structured_schema = {
            "node_props": {el[0]["label"]: el[0]["keys"] for el in node_properties},
            "rel_props": {el[0]["types"]: el[0]["keys"] for el in rel_properties},
            "relationships": [dict(el[0]) for el in relationships],
        }

        self.schema = (
            f"Node properties: {node_properties}\n"
            f"Relationships properties: {rel_properties}\n"
            f"Relationships: {relationships}\n"
        )

    def query(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        """Query the FalkorDB database.

        Args:
            query: The Cypher query to execute.
            params: Dictionary of query parameters.

        Returns:
            The query result set as a list of rows.
        """
        try:
            data = self._graph.query(query, params or {})
            return data.result_set
        except Exception as e:
            raise ValueError(f"Generated Cypher Statement is not valid\n{e}") from e

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Construct nodes and relationships in the graph from documents.

        Args:
            graph_documents: The graph documents to ingest.
            include_source: If ``True``, also create a ``Document`` node per
                graph document source and link every entity node to it with
                a ``MENTIONS`` relationship.
        """
        for document in graph_documents:
            if include_source and document.source is None:
                raise ValueError(
                    "include_source is set to True, but a graph document has no source."
                )

            # Import nodes. Ids and properties are passed as parameters,
            # never interpolated into the query.
            for node in document.nodes:
                label = _validated_label(node.type)
                self.query(
                    f"MERGE (n:`{label}` {{id: $id}}) SET n += $properties",
                    {"id": node.id, "properties": node.properties or {}},
                )

            # Import relationships
            for rel in document.relationships:
                source_label = _validated_label(rel.source.type)
                target_label = _validated_label(rel.target.type)
                rel_type = _validated_label(rel.type.replace(" ", "_").upper())
                self.query(
                    f"MATCH (a:`{source_label}` {{id: $source_id}}), "
                    f"(b:`{target_label}` {{id: $target_id}}) "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) "
                    "SET r += $properties",
                    {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id,
                        "properties": rel.properties or {},
                    },
                )

            # Link entity nodes to their source document
            if include_source and document.source is not None:
                source = document.source
                source_id = (
                    source.id or md5(source.page_content.encode("utf-8")).hexdigest()
                )
                self.query(
                    "MERGE (d:Document {id: $id}) "
                    "SET d.text = $text SET d += $metadata",
                    {
                        "id": source_id,
                        "text": source.page_content,
                        "metadata": source.metadata or {},
                    },
                )
                for node in document.nodes:
                    label = _validated_label(node.type)
                    self.query(
                        "MATCH (d:Document {id: $source_id}) "
                        f"MATCH (n:`{label}` {{id: $node_id}}) "
                        "MERGE (d)-[:MENTIONS]->(n)",
                        {"source_id": source_id, "node_id": node.id},
                    )
