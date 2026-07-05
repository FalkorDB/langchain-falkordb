"""Integration tests for the FalkorDB vector store.

These tests require a running FalkorDB instance. By default they connect to
``localhost:6379``; set ``FALKORDB_HOST`` / ``FALKORDB_PORT`` to override
(e.g. for a FalkorDB Cloud instance).

Run FalkorDB locally with:

.. code-block:: bash

    docker run -p 6379:6379 -it --rm falkordb/falkordb:latest
"""

import os
from typing import Generator, List

import pytest
from falkordb import FalkorDB
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import ReadWriteTestSuite

from langchain_falkordb.vectorstores import FalkorDBVector, SearchType

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", 6379))


def drop_graph(store: FalkorDBVector) -> None:
    """Delete the graph backing a store, ignoring 'graph never created'."""
    try:
        store._database.delete()
    except Exception:
        pass


class TestFalkorDBStandard(ReadWriteTestSuite):
    """LangChain standard read/write test suite for FalkorDBVector."""

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore backed by a fresh random graph."""
        store = FalkorDBVector(
            embedding=self.get_embeddings(),
            host=host,
            port=port,
        )
        try:
            yield store
        finally:
            drop_graph(store)


class FixedVectorEmbeddings(Embeddings):
    """Deterministic embeddings with controlled geometry (4 dimensions)."""

    vectors = {
        "apple": [1.0, 0.0, 0.0, 0.0],
        "orange": [0.9, 0.1, 0.0, 0.0],
        "banana": [0.0, 1.0, 0.0, 0.0],
        "car": [0.0, 0.0, 1.0, 0.0],
        "boat": [0.0, 0.0, 0.9, 0.1],
    }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # The store probes embed_query with arbitrary text to detect the
        # embedding dimension, so unknown texts get a neutral vector.
        return self.vectors.get(text, [0.5, 0.5, 0.5, 0.5])


ALL_TEXTS = ["apple", "orange", "banana", "car", "boat"]


@pytest.fixture()
def store() -> Generator[FalkorDBVector, None, None]:
    vectorstore = FalkorDBVector(
        embedding=FixedVectorEmbeddings(),
        host=host,
        port=port,
    )
    try:
        yield vectorstore
    finally:
        drop_graph(vectorstore)


def test_similarity_search_respects_k(store: FalkorDBVector) -> None:
    """similarity_search(k=2) must return exactly the 2 nearest documents."""
    store.add_texts(ALL_TEXTS)
    output = store.similarity_search("apple", k=2)
    assert [doc.page_content for doc in output] == ["apple", "orange"]


def test_similarity_search_orders_by_distance(store: FalkorDBVector) -> None:
    store.add_texts(["banana", "apple", "orange"])
    output = store.similarity_search_with_score("apple", k=3)
    assert [doc.page_content for doc, _ in output] == ["apple", "orange", "banana"]
    scores = [score for _, score in output]
    assert scores == sorted(scores)


def test_random_database_names_are_unique() -> None:
    """Two stores created without a database name must not share a graph."""
    store_one = FalkorDBVector(embedding=FixedVectorEmbeddings(), host=host, port=port)
    store_two = FalkorDBVector(embedding=FixedVectorEmbeddings(), host=host, port=port)
    try:
        assert store_one.database_name != store_two.database_name
    finally:
        drop_graph(store_one)
        drop_graph(store_two)


def test_explicit_embedding_dimension() -> None:
    """Passing embedding_dimension explicitly must be honored."""
    vectorstore = FalkorDBVector(
        embedding=FixedVectorEmbeddings(),
        host=host,
        port=port,
        embedding_dimension=4,
    )
    try:
        assert vectorstore.embedding_dimension == 4
        vectorstore.add_texts(["apple"])
        output = vectorstore.similarity_search("apple", k=1)
        assert [doc.page_content for doc in output] == ["apple"]
    finally:
        drop_graph(vectorstore)


def test_similarity_search_by_vector_without_query(store: FalkorDBVector) -> None:
    """similarity_search_by_vector must work without a `query` kwarg."""
    store.add_texts(ALL_TEXTS)
    embedding = FixedVectorEmbeddings().embed_query("banana")
    output = store.similarity_search_by_vector(embedding, k=1)
    assert [doc.page_content for doc in output] == ["banana"]


def test_metadata_filter(store: FalkorDBVector) -> None:
    store.add_texts(
        ALL_TEXTS,
        metadatas=[
            {"kind": "fruit"},
            {"kind": "fruit"},
            {"kind": "fruit"},
            {"kind": "vehicle"},
            {"kind": "vehicle"},
        ],
    )
    output = store.similarity_search("apple", k=4, filter={"kind": "vehicle"})
    assert {doc.page_content for doc in output} == {"car", "boat"}
    assert all(doc.metadata["kind"] == "vehicle" for doc in output)


def test_metadata_filter_respects_k(store: FalkorDBVector) -> None:
    store.add_texts(ALL_TEXTS, metadatas=[{"kind": "thing"} for _ in ALL_TEXTS])
    output = store.similarity_search("apple", k=2, filter={"kind": "thing"})
    assert len(output) == 2


def test_metadata_roundtrip_with_reserved_id_key(store: FalkorDBVector) -> None:
    """Metadata with an `id` key must not clobber the document id."""
    store.add_texts(["apple"], metadatas=[{"id": 7, "source": "s1"}], ids=["docA"])
    document = store.similarity_search("apple", k=1)[0]
    assert document.id == "docA"
    assert document.metadata == {"id": 7, "source": "s1"}


def test_max_marginal_relevance_search(store: FalkorDBVector) -> None:
    store.add_texts(ALL_TEXTS)
    output = store.max_marginal_relevance_search("apple", k=2, fetch_k=4)
    assert len(output) == 2
    assert output[0].page_content == "apple"
    assert all("embedding" not in doc.metadata for doc in output)


def test_construct_with_graph_object() -> None:
    """A FalkorDBGraph-like object (duck-typed) can supply the connection."""
    driver = FalkorDB(host=host, port=port)

    class DuckGraph:
        def __init__(self) -> None:
            self._driver = driver
            self._graph = driver.select_graph("duck_graph_test")

    vectorstore = FalkorDBVector(embedding=FixedVectorEmbeddings(), graph=DuckGraph())
    try:
        vectorstore.add_texts(["apple", "banana"])
        output = vectorstore.similarity_search("banana", k=1)
        assert [doc.page_content for doc in output] == ["banana"]
    finally:
        drop_graph(vectorstore)


def test_hybrid_search() -> None:
    vectorstore = FalkorDBVector.from_texts(
        texts=ALL_TEXTS,
        embedding=FixedVectorEmbeddings(),
        host=host,
        port=port,
        search_type=SearchType.HYBRID,
    )
    try:
        output = vectorstore.similarity_search("apple", k=2)
        assert len(output) == 2
        assert output[0].page_content == "apple"
    finally:
        drop_graph(vectorstore)


def test_from_existing_index() -> None:
    database = "existing_index_test"
    original = FalkorDBVector.from_texts(
        texts=["apple", "banana"],
        embedding=FixedVectorEmbeddings(),
        host=host,
        port=port,
        database=database,
    )
    try:
        reconnected = FalkorDBVector.from_existing_index(
            embedding=FixedVectorEmbeddings(),
            node_label="Chunk",
            host=host,
            port=port,
            database=database,
        )
        output = reconnected.similarity_search("banana", k=1)
        assert [doc.page_content for doc in output] == ["banana"]
    finally:
        drop_graph(original)


def test_update_documents(store: FalkorDBVector) -> None:
    """update_documents must re-embed the new content, not just the text."""
    from langchain_core.documents import Document

    store.add_texts(["apple", "boat"], ids=["doc1", "doc2"])
    store.update_documents(
        "doc1", Document(page_content="car", metadata={"kind": "vehicle"})
    )
    # If the embedding were stale ("apple"), the nearest match for "car"
    # would be "boat", not the updated document.
    document = store.similarity_search("car", k=1)[0]
    assert document.id == "doc1"
    assert document.page_content == "car"
    assert document.metadata["kind"] == "vehicle"
    with pytest.raises(ValueError, match="not found"):
        store.update_documents("missing", Document(page_content="apple"))


def test_relationship_index_search() -> None:
    """Vector search over an externally created relationship index."""
    driver = FalkorDB(host=host, port=port)
    graph = driver.select_graph("rel_index_test")
    try:
        graph.create_edge_vector_index(
            "LINKS", "embedding", dim=4, similarity_function="cosine"
        )
        graph.query(
            "CREATE (:A {name:'a1'})-[:LINKS {text:'apple', "
            "embedding: vecf32([1.0,0.0,0.0,0.0])}]->(:B {name:'b1'})"
        )
        graph.query(
            "CREATE (:A {name:'a2'})-[:LINKS {text:'banana', "
            "embedding: vecf32([0.0,1.0,0.0,0.0])}]->(:B {name:'b2'})"
        )
        vectorstore = FalkorDBVector.from_existing_relationship_index(
            embedding=FixedVectorEmbeddings(),
            relation_type="LINKS",
            host=host,
            port=port,
            database="rel_index_test",
        )
        output = vectorstore.similarity_search("apple", k=1)
        assert [doc.page_content for doc in output] == ["apple"]
    finally:
        try:
            graph.delete()
        except Exception:
            pass


def test_delete_scoped_to_store_label(store: FalkorDBVector) -> None:
    """delete() without ids must clear the store, not unrelated nodes."""
    store.add_texts(["apple"])
    store._database.query("CREATE (:Unrelated {name: 'keepme'})")
    store.delete()
    assert store.similarity_search("apple", k=1) == []
    remaining = store._query("MATCH (n:Unrelated) RETURN count(n)")
    assert remaining[0][0] == 1
