"""Unit tests for pure helpers in langchain_falkordb.vectorstores."""

from typing import List

import pytest
from langchain_core.embeddings import Embeddings

from langchain_falkordb.vectorstores import (
    FalkorDBVector,
    IndexType,
    SearchType,
    _get_search_index_query,
    construct_metadata_filter,
    dict_to_yaml_str,
    process_index_data,
)


class NoopEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0]


class TestIdentifierValidation:
    """Backticks in identifiers would escape the quoted Cypher identifier."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"node_label": "Chunk`) DELETE (n"},
            {"relation_type": "LINKS`) DELETE (n"},
            {"embedding_node_property": "embedding`) DELETE (n"},
            {"text_node_property": "text`) DELETE (n"},
        ],
    )
    def test_backtick_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="backtick"):
            FalkorDBVector(embedding=NoopEmbeddings(), **kwargs)

    @pytest.mark.parametrize(
        "bad_property",
        ["bad`prop", "bad'prop", 'bad"prop', "bad\\prop"],
    )
    def test_from_existing_graph_rejects_unsafe_properties(
        self, bad_property: str
    ) -> None:
        """Text property names are interpolated into Cypher and must be safe."""
        with pytest.raises(ValueError, match="text_node_properties"):
            FalkorDBVector.from_existing_graph(
                embedding=NoopEmbeddings(),
                database="db",
                node_label="Node",
                embedding_node_property="embedding",
                text_node_properties=[bad_property],
            )


class TestConstructMetadataFilter:
    def test_empty_filter(self) -> None:
        assert construct_metadata_filter(None) == ("", {})
        assert construct_metadata_filter({}) == ("", {})

    def test_values_are_parameterized(self) -> None:
        """Filter values must be passed as query parameters, not inlined."""
        snippet, params = construct_metadata_filter({"page": "0"})
        assert snippet == "n.`page` = $filter_param_0"
        assert params == {"filter_param_0": "0"}

    def test_multiple_conditions(self) -> None:
        snippet, params = construct_metadata_filter({"page": "0", "count": 3})
        assert snippet == "n.`page` = $filter_param_0 AND n.`count` = $filter_param_1"
        assert params == {"filter_param_0": "0", "filter_param_1": 3}

    def test_invalid_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            construct_metadata_filter({"bad`key": 1})
        with pytest.raises(ValueError):
            construct_metadata_filter({"a b": 1})


class TestProcessIndexData:
    # Shape matches a real `CALL db.indexes()` row from FalkorDB:
    # [label, properties, types, options, language, stopwords,
    #  entitytype, status, info]
    ROW = [
        "Chunk",
        ["embedding", "text"],
        {"embedding": ["VECTOR"], "text": ["FULLTEXT"]},
        {
            "embedding": {
                "dimension": 4,
                "similarityFunction": "cosine",
                "M": 16,
                "efConstruction": 200,
                "efRuntime": 10,
            },
            "text": {},
        },
        "english",
        [],
        "NODE",
        "OPERATIONAL",
        {},
    ]

    def test_vector_index_entry(self) -> None:
        entries = process_index_data([self.ROW])
        vector = [e for e in entries if e["index_type"] == "VECTOR"]
        assert vector == [
            {
                "entity_label": "Chunk",
                "entity_property": "embedding",
                "entity_type": "NODE",
                "index_type": "VECTOR",
                "index_status": "OPERATIONAL",
                "index_dimension": 4,
                "index_similarityFunction": "cosine",
            }
        ]

    def test_fulltext_index_entry(self) -> None:
        entries = process_index_data([self.ROW])
        fulltext = [e for e in entries if e["index_type"] == "FULLTEXT"]
        assert len(fulltext) == 1
        assert fulltext[0]["entity_property"] == "text"
        assert fulltext[0]["index_dimension"] is None


class TestSearchIndexQuery:
    def test_node_vector_query(self) -> None:
        query = _get_search_index_query(SearchType.VECTOR, IndexType.NODE)
        assert "db.idx.vector.queryNodes" in query

    def test_node_hybrid_query(self) -> None:
        query = _get_search_index_query(SearchType.HYBRID, IndexType.NODE)
        assert "db.idx.vector.queryNodes" in query
        assert "db.idx.fulltext.queryNodes" in query

    def test_relationship_vector_query(self) -> None:
        query = _get_search_index_query(SearchType.VECTOR, IndexType.RELATIONSHIP)
        assert "db.idx.vector.queryRelationships" in query

    def test_relationship_hybrid_unsupported(self) -> None:
        with pytest.raises(ValueError):
            _get_search_index_query(SearchType.HYBRID, IndexType.RELATIONSHIP)


def test_dict_to_yaml_str() -> None:
    assert dict_to_yaml_str({"a": 1}) == "a: 1\n"
    assert dict_to_yaml_str({"a": {"b": "c"}}) == "a:\n  b: c\n"
    assert dict_to_yaml_str({"a": [1, 2]}) == "a:\n- 1\n- 2\n"


def test_enums() -> None:
    assert SearchType.VECTOR == "vector"
    assert SearchType.HYBRID == "hybrid"
    assert IndexType.NODE == "NODE"
    assert IndexType.RELATIONSHIP == "RELATIONSHIP"
