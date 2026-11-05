"""Minimal FalkorDB + LangChain integration demo."""

from __future__ import annotations

import argparse
import logging

from langchain_core.documents import Document

from langchain_falkordb import FalkorDBGraph
from langchain_falkordb.graphs import GraphDocument, Node, Relationship


def build_demo_graph(graph: FalkorDBGraph) -> None:
    """Insert a tiny graph into FalkorDB."""
    source = Document(page_content="Tom Hanks acted in Forrest Gump.")
    tom = Node(id="Tom Hanks", type="Actor")
    gump = Node(id="Forrest Gump", type="Movie")
    graph.add_graph_documents(
        [
            GraphDocument(
                nodes=[tom, gump],
                relationships=[Relationship(source=tom, target=gump, type="ACTED_IN")],
                source=source,
            )
        ],
        include_source=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=6379, type=int)
    parser.add_argument("--database", default="langchain_falkordb_demo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    graph = FalkorDBGraph(args.database, host=args.host, port=args.port)
    build_demo_graph(graph)
    graph.refresh_schema()
    rows = graph.query("MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.id, m.id")

    logging.info("Schema:\n%s", graph.get_schema)
    logging.info("Query result: %s", rows)


if __name__ == "__main__":
    main()
