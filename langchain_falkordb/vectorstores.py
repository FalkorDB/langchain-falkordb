"""FalkorDB vector store integration for LangChain."""

from __future__ import annotations

import enum
import logging
import os
import re
import uuid
from hashlib import md5
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_falkordb.utils import DistanceStrategy, maximal_marginal_relevance

logger = logging.getLogger(__name__)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "euclidean",
    DistanceStrategy.COSINE: "cosine",
}

# Node property that stores the LangChain document id. It is deliberately
# not called "id" so that user metadata keys (including "id") never collide
# with it.
DOC_ID_PROPERTY = "_id"

# Error raised by FalkorDB when a vector index (or the label/property it
# refers to) does not exist in the graph.
_MISSING_INDEX_ERROR = "Attempted to access undefined attribute"

# FalkorDB's HNSW index keeps tombstones for updated/deleted entries and
# counts them toward the requested k of db.idx.vector.query*, so fewer than
# k live results can come back even when more exist. Searches therefore
# over-fetch with escalation, capped here.
_MAX_FETCH_K = 4096

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SearchType(str, enum.Enum):
    """Enumerator of the search strategies of the FalkorDB vector store.

    - ``SearchType.VECTOR`` searches using only the vector index, relying
      on the similarity between vector embeddings to return relevant
      results.
    - ``SearchType.HYBRID`` performs a combined search, querying both the
      full-text index and the vector index, and merges the normalized
      results.
    """

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR


class IndexType(str, enum.Enum):
    """Enumerator of the index types."""

    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"


DEFAULT_INDEX_TYPE = IndexType.NODE


def _generate_database_name() -> str:
    """Generate a unique name for a new graph."""
    return f"vector_store_{uuid.uuid4().hex}"


def dict_to_yaml_str(input_dict: Dict, indent: int = 0) -> str:
    """Convert a dictionary to a YAML-like string without external libraries.

    Args:
        input_dict: The dictionary to convert.
        indent: The current indentation level.

    Returns:
        The YAML-like string representation of the input dictionary.
    """
    yaml_str = ""
    for key, value in input_dict.items():
        padding = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{padding}{key}:\n{dict_to_yaml_str(value, indent + 1)}"
        elif isinstance(value, list):
            yaml_str += f"{padding}{key}:\n"
            for item in value:
                yaml_str += f"{padding}- {item}\n"
        else:
            yaml_str += f"{padding}{key}: {value}\n"
    return yaml_str


def construct_metadata_filter(
    filter: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Construct a Cypher filter snippet from a metadata filter dict.

    Filter values are passed as query parameters (never inlined) to avoid
    Cypher injection.

    Args:
        filter: Dictionary of equality conditions on metadata properties.

    Returns:
        A tuple of the filter snippet and the query parameters it uses.
    """
    if not filter:
        return "", {}

    snippets = []
    params: Dict[str, Any] = {}
    for i, (key, value) in enumerate(filter.items()):
        if not _IDENTIFIER_PATTERN.match(key):
            raise ValueError(
                f"Invalid metadata filter key: {key!r}. Filter keys must be "
                "valid identifiers (letters, digits and underscores, not "
                "starting with a digit)."
            )
        param_name = f"filter_param_{i}"
        snippets.append(f"n.`{key}` = ${param_name}")
        params[param_name] = value

    return " AND ".join(snippets), params


def _get_search_index_query(
    search_type: SearchType, index_type: IndexType = DEFAULT_INDEX_TYPE
) -> str:
    """Build the index-query part of a similarity search Cypher query."""
    if index_type == IndexType.NODE:
        if search_type == SearchType.VECTOR:
            return (
                "CALL db.idx.vector.queryNodes($entity_label, "
                "$entity_property, $fetch_k, vecf32($embedding)) "
                "YIELD node, score "
                "WITH node, score ORDER BY score ASC "
            )
        # Hybrid search. FalkorDB vector scores are distances (lower is
        # better) while full-text scores are similarities (higher is
        # better), so the vector scores are converted before both are
        # normalized to [0, 1] and merged.
        return (
            "CALL { "
            "CALL db.idx.vector.queryNodes($entity_label, "
            "$entity_property, $fetch_k, vecf32($embedding)) "
            "YIELD node, score "
            "WITH collect({node: node, score: score})"
            " AS nodes, max(score) AS max_score "
            "UNWIND nodes AS n "
            "RETURN n.node AS node, "
            "(1 - (n.score / (max_score + 0.0001))) AS score "
            "UNION "
            "CALL db.idx.fulltext.queryNodes($entity_label, $query) "
            "YIELD node, score "
            "WITH collect({node: node, score: score})"
            " AS nodes, max(score) AS max_score "
            "UNWIND nodes AS n "
            "RETURN n.node AS node, (n.score / max_score) AS score "
            "} "
            "WITH node, max(score) AS score "
            "ORDER BY score DESC LIMIT $k "
        )
    if index_type == IndexType.RELATIONSHIP:
        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector indexes"
            )
        return (
            "CALL db.idx.vector.queryRelationships"
            "($entity_label, $entity_property, $fetch_k, vecf32($embedding)) "
            "YIELD relationship, score "
            "WITH relationship, score ORDER BY score ASC "
        )
    raise ValueError(f"Unsupported index type: {index_type}")


def process_index_data(data: List[List[Any]]) -> List[Dict[str, Any]]:
    """Process the output of ``CALL db.indexes()`` into flat dictionaries.

    Args:
        data: The raw result set of ``CALL db.indexes()``. Each row has the
            shape ``[label, properties, types, options, language, stopwords,
            entitytype, status, info]``.

    Returns:
        One dictionary per indexed property with the keys:

        - ``entity_label``: label of the indexed node or relationship.
        - ``entity_property``: property the index was created on.
        - ``entity_type``: ``"NODE"`` or ``"RELATIONSHIP"``.
        - ``index_type``: e.g. ``"VECTOR"`` or ``"FULLTEXT"``.
        - ``index_status``: e.g. ``"OPERATIONAL"``.
        - ``index_dimension``: vector index dimension, if applicable.
        - ``index_similarityFunction``: vector similarity function, if
          applicable.
    """
    result = []

    for entity in data:
        entity_label = entity[0]
        index_type_map = entity[2]
        index_options = entity[3]
        entity_type = entity[6]
        index_status = entity[7]

        for prop, index_types in index_type_map.items():
            entity_info: Dict[str, Any] = {
                "entity_label": entity_label,
                "entity_property": prop,
                "entity_type": entity_type,
                "index_type": index_types[0],
                "index_status": index_status,
                "index_dimension": None,
                "index_similarityFunction": None,
            }

            if "VECTOR" in index_types and isinstance(index_options, dict):
                vector_info = index_options.get(prop, {})
                entity_info["index_dimension"] = vector_info.get("dimension")
                entity_info["index_similarityFunction"] = vector_info.get(
                    "similarityFunction"
                )

            result.append(entity_info)

    return result


class FalkorDBVector(VectorStore):
    """FalkorDB vector store.

    To use, you should have the ``falkordb`` python package installed and a
    running FalkorDB instance, e.g. via
    ``docker run -p 6379:6379 falkordb/falkordb``.

    Args:
        embedding: Any embedding function implementing the
            ``langchain_core.embeddings.Embeddings`` interface.
        host: FalkorDB host. Defaults to ``"localhost"``.
        port: FalkorDB port. Defaults to ``6379``.
        username: Optionally provide your username if you are connecting to
            a protected (e.g. FalkorDB Cloud) database instance. Falls back
            to the ``FALKORDB_USERNAME`` environment variable.
        password: Optionally provide your password if you are connecting to
            a protected (e.g. FalkorDB Cloud) database instance. Falls back
            to the ``FALKORDB_PASSWORD`` environment variable.
        ssl: Whether the connection should use SSL/TLS encryption.
            Defaults to ``False``.
        database: Name of the graph to store the data in. A unique name is
            generated if not provided.
        node_label: Label of the nodes the documents are stored in.
            Defaults to ``"Chunk"``.
        relation_type: Relationship type used when querying a relationship
            vector index. Defaults to ``""``.
        embedding_node_property: Name of the property the embeddings are
            stored in. Defaults to ``"embedding"``.
        text_node_property: Name of the property the document text is
            stored in. Defaults to ``"text"``.
        embedding_dimension: Dimension of the embeddings. Calculated from
            the embedding function if not provided.
        distance_strategy: The distance strategy to use
            (``COSINE`` or ``EUCLIDEAN_DISTANCE``). Defaults to ``COSINE``.
        search_type: Similarity search type to use, either
            ``SearchType.VECTOR`` or ``SearchType.HYBRID``.
            Defaults to ``SearchType.VECTOR``.
        retrieval_query: Optional custom Cypher snippet appended to the
            index query. It must return the columns ``text``, ``score`` and
            ``metadata``.
        index_type: Type of the vector index (node or relationship).
            Defaults to ``IndexType.NODE``.
        graph: Optional graph object to reuse an existing connection. Any
            object exposing ``_driver`` (a ``falkordb.FalkorDB``) and
            ``_graph`` (a ``falkordb.Graph``) attribute is accepted, e.g. a
            ``FalkorDBGraph`` instance from ``langchain-community``.
        relevance_score_fn: Optional function to override the relevance
            score computed from the similarity score.
        pre_delete_collection: If ``True``, delete existing data under
            ``node_label`` on initialization. Useful for testing.

    Example:
        .. code-block:: python

            from langchain_falkordb import FalkorDBVector
            from langchain_openai import OpenAIEmbeddings

            vectorstore = FalkorDBVector.from_texts(
                texts=["hello world"],
                embedding=OpenAIEmbeddings(),
                host="localhost",
                port=6379,
            )
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        database: Optional[str] = None,
        node_label: str = "Chunk",
        relation_type: str = "",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        embedding_dimension: Optional[int] = None,
        retrieval_query: str = "",
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        graph: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        ssl: bool = False,
        pre_delete_collection: bool = False,
    ) -> None:
        try:
            import falkordb
        except ImportError as e:
            raise ImportError(
                "Could not import falkordb python package. "
                "Please install it with `pip install falkordb`."
            ) from e

        # Allow only cosine and euclidean distance strategies
        if distance_strategy not in [
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.COSINE,
        ]:
            raise ValueError(
                "`distance_strategy` must be either `EUCLIDEAN_DISTANCE` or `COSINE`"
            )
        if not embedding_node_property:
            raise ValueError(
                "The `embedding_node_property` must not be None or empty string"
            )
        if not node_label:
            raise ValueError("The `node_label` must not be None or empty string")

        if graph is not None:
            # Reuse the connection of a FalkorDBGraph-like object.
            self._driver = graph._driver
            self._database = graph._graph
            self.database_name: str = self._database.name
        else:
            self._host = host
            self._port = port
            self._username = username or os.environ.get("FALKORDB_USERNAME")
            self._password = password or os.environ.get("FALKORDB_PASSWORD")
            self._ssl = ssl

            try:
                self._driver = falkordb.FalkorDB(
                    host=self._host,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                    ssl=self._ssl,
                )
            except Exception as e:
                raise ValueError(
                    "Could not connect to FalkorDB database. "
                    "Please ensure that the host, port, username, "
                    "and password are correct."
                ) from e

            self.database_name = database or _generate_database_name()
            self._database = self._driver.select_graph(self.database_name)

        self.embedding = embedding
        self.node_label = node_label
        self.relation_type = relation_type
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self._distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self.pre_delete_collection = pre_delete_collection
        self.retrieval_query = retrieval_query
        self.search_type = search_type
        self._index_type = index_type
        self._index_verified = False

        # Calculate embedding dimension if not given
        self.embedding_dimension = embedding_dimension or len(
            self.embedding.embed_query("foo")
        )

        # Delete existing data if flagged
        if pre_delete_collection:
            self._database.query(f"MATCH (n:`{self.node_label}`) DELETE n")

    @property
    def embeddings(self) -> Embeddings:
        """Return the ``Embeddings`` model used by the vector store."""
        return self.embedding

    def _query(
        self,
        query: str,
        *,
        params: Optional[dict] = None,
        retry_on_timeout: bool = True,
    ) -> List[List[Any]]:
        """Send a Cypher query to the FalkorDB database.

        Args:
            query: The Cypher query to execute.
            params: Dictionary of query parameters. Defaults to ``{}``.

        Returns:
            The query result set as a list of rows.
        """
        params = params or {}
        try:
            data = self._database.query(query, params)
            return data.result_set
        except Exception as e:
            if "Invalid input" in str(e):
                raise ValueError(f"Cypher Statement is not valid\n{e}") from e
            if retry_on_timeout:
                return self._query(query, params=params, retry_on_timeout=False)
            raise

    def retrieve_existing_node_index(
        self, node_label: Optional[str] = ""
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Check if a node vector index exists in the FalkorDB database.

        Args:
            node_label: Node label to look for. Defaults to the label the
                store was configured with.

        Returns:
            A tuple of the embedding dimension, the entity type, the entity
            label and the indexed property of the existing vector index, or
            ``(None, None, None, None)`` if no matching index exists.
        """
        node_label = node_label or self.node_label
        if not node_label:
            raise ValueError("`node_label` property must be set to use this function")

        index_information = self._database.query("CALL db.indexes()")
        if not index_information:
            return None, None, None, None

        for index in process_index_data(index_information.result_set):
            if (
                index.get("entity_label") == node_label
                and index.get("entity_type") == "NODE"
                and index.get("index_type") == "VECTOR"
                and index.get("index_dimension") is not None
            ):
                self._index_type = IndexType(index["entity_type"])
                return (
                    int(index["index_dimension"]),
                    str(index["entity_type"]),
                    str(index["entity_label"]),
                    str(index["entity_property"]),
                )
        return None, None, None, None

    def retrieve_existing_relationship_index(
        self, relation_type: Optional[str] = ""
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Check if a relationship vector index exists in the database.

        Args:
            relation_type: Relationship type to look for. Defaults to the
                relation type the store was configured with.

        Returns:
            A tuple of the embedding dimension, the entity type, the entity
            label and the indexed property of the existing vector index, or
            ``(None, None, None, None)`` if no matching index exists.
        """
        relation_type = relation_type or self.relation_type
        if not relation_type:
            raise ValueError(
                "Couldn't find any specified `relation_type`. "
                "Check if you spelled it correctly"
            )

        index_information = self._database.query("CALL db.indexes()")
        if not index_information:
            return None, None, None, None

        for index in process_index_data(index_information.result_set):
            if (
                index.get("entity_label") == relation_type
                and index.get("entity_type") == "RELATIONSHIP"
                and index.get("index_type") == "VECTOR"
                and index.get("index_dimension") is not None
            ):
                self._index_type = IndexType(index["entity_type"])
                return (
                    int(index["index_dimension"]),
                    str(index["entity_type"]),
                    str(index["entity_label"]),
                    str(index["entity_property"]),
                )
        return None, None, None, None

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """Check if a full-text index exists for the store's node label.

        Returns:
            The entity label of the full-text index, or ``None`` if it does
            not exist.
        """
        index_information = self._database.query("CALL db.indexes()")
        if not index_information:
            return None

        for index in process_index_data(index_information.result_set):
            if (
                index.get("entity_label") == self.node_label
                and index.get("index_type") == "FULLTEXT"
            ):
                return str(index["entity_label"])
        return None

    def create_new_node_index(
        self,
        node_label: Optional[str] = "",
        embedding_node_property: Optional[str] = "",
        embedding_dimension: Optional[int] = None,
    ) -> None:
        """Create a new vector index on a node label in FalkorDB.

        All arguments default to the values the store was configured with.
        """
        node_label = node_label or self.node_label
        embedding_node_property = (
            embedding_node_property or self.embedding_node_property
        )
        embedding_dimension = embedding_dimension or self.embedding_dimension
        if not node_label:
            raise ValueError("`node_label` property must be set to use this function")
        if not embedding_node_property:
            raise ValueError(
                "`embedding_node_property` property must be set to use this function"
            )
        if not embedding_dimension:
            raise ValueError(
                "`embedding_dimension` property must be set to use this function"
            )

        try:
            self._database.create_node_vector_index(
                node_label,
                embedding_node_property,
                dim=embedding_dimension,
                similarity_function=DISTANCE_MAPPING[self._distance_strategy],
            )
        except Exception as e:
            if "already indexed" in str(e):
                raise ValueError(
                    f"A vector index on (:{node_label}"
                    "{"
                    f"{embedding_node_property}"
                    "}) has already been created"
                ) from e
            raise

    def create_new_index_on_relationship(
        self,
        relation_type: str = "",
        embedding_node_property: str = "",
        embedding_dimension: int = 0,
    ) -> None:
        """Create a new vector index on a relationship type in FalkorDB.

        All arguments default to the values the store was configured with.
        """
        relation_type = relation_type or self.relation_type
        embedding_node_property = (
            embedding_node_property or self.embedding_node_property
        )
        embedding_dimension = embedding_dimension or self.embedding_dimension
        if not relation_type:
            raise ValueError("`relation_type` must be set to use this function")
        if not embedding_node_property:
            raise ValueError(
                "`embedding_node_property` must be set to use this function"
            )
        if not embedding_dimension:
            raise ValueError("`embedding_dimension` must be set to use this function")

        try:
            self._database.create_edge_vector_index(
                relation_type,
                embedding_node_property,
                dim=embedding_dimension,
                similarity_function=DISTANCE_MAPPING[self._distance_strategy],
            )
        except Exception as e:
            if "already indexed" in str(e):
                raise ValueError(
                    f"A vector index on [:{relation_type}"
                    "{"
                    f"{embedding_node_property}"
                    "}] has already been created"
                ) from e
            raise

    def create_new_keyword_index(
        self, text_node_properties: Optional[List[str]] = None
    ) -> None:
        """Create a new full-text index in FalkorDB.

        Args:
            text_node_properties: List of node properties to be indexed.
                If not provided, defaults to the store's
                ``text_node_property``.
        """
        node_props = text_node_properties or [self.text_node_property]
        self._database.create_node_fulltext_index(self.node_label, *node_props)

    def _ensure_node_index(self) -> None:
        """Create the node vector index if missing, or validate it."""
        if self._index_verified:
            return

        embedding_dimension, index_type, entity_label, entity_property = (
            self.retrieve_existing_node_index()
        )
        if not index_type:
            self.create_new_node_index()
        elif (
            embedding_dimension is not None
            and self.embedding_dimension != embedding_dimension
        ):
            raise ValueError(
                f"A vector index for {entity_label} on {entity_property} "
                "exists, but the provided embedding function and vector "
                "index dimensions do not match.\n"
                f"Embedding function dimension: {self.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        self._index_verified = True

    def _ensure_fts_index(
        self, text_node_properties: Optional[List[str]] = None
    ) -> None:
        """Create the full-text index if missing, or validate it."""
        fts_node_label = self.retrieve_existing_fts_index()
        if not fts_node_label:
            self.create_new_keyword_index(text_node_properties)
        elif fts_node_label != self.node_label:
            raise ValueError("Vector and keyword index don't index the same node label")

    @staticmethod
    def _coerce_ids(
        texts: List[str], ids: Optional[Sequence[Optional[str]]]
    ) -> List[str]:
        """Fill in missing document ids with deterministic text hashes."""
        if ids is None:
            return [md5(text.encode("utf-8")).hexdigest() for text in texts]
        if len(ids) != len(texts):
            raise ValueError("The number of ids must match the number of texts.")
        return [
            str(id_) if id_ is not None else md5(text.encode("utf-8")).hexdigest()
            for id_, text in zip(ids, texts)
        ]

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts with pre-generated embeddings to the vector store.

        Documents are upserted by id: adding a document with an existing id
        overwrites the stored text and metadata.

        Args:
            texts: Iterable of strings to add to the vector store.
            embeddings: List of embedding vectors, one per text.
            metadatas: Optional list of metadata dicts associated with the
                texts. Metadata values must be primitive types supported as
                FalkorDB properties.
            ids: Optional list of document ids. Missing entries are filled
                with a hash of the text.

        Returns:
            The list of ids of the added texts.
        """
        if self._index_type == IndexType.RELATIONSHIP:
            raise ValueError(
                "Data ingestion is not supported with relationship vector index"
            )

        texts = list(texts)
        coerced_ids = self._coerce_ids(texts, ids)
        if not metadatas:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("The number of metadatas must match the number of texts.")

        rows = []
        for text, metadata, embedding, id_ in zip(
            texts, metadatas, embeddings, coerced_ids
        ):
            properties = dict(metadata)
            properties[DOC_ID_PROPERTY] = id_
            properties[self.text_node_property] = text
            rows.append({"id": id_, "props": properties, "embedding": embedding})

        self._ensure_node_index()
        self._database.query(
            "UNWIND $data AS row "
            f"MERGE (c:`{self.node_label}` {{`{DOC_ID_PROPERTY}`: row.id}}) "
            "SET c = row.props, "
            f"c.`{self.embedding_node_property}` = vecf32(row.embedding)",
            {"data": rows},
        )
        return coerced_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through the embeddings and add them to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadata dicts associated with the
                texts.
            ids: Optional list of document ids. Missing entries are filled
                with a hash of the text.

        Returns:
            The list of ids of the added texts.
        """
        texts = list(texts)
        embeddings = self.embedding.embed_documents(texts)
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to retrieve.

        Returns:
            The found documents, in the same order as the requested ids.
            Missing ids are skipped.
        """
        rows = self._query(
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{DOC_ID_PROPERTY}` IN $ids "
            f"RETURN n.`{DOC_ID_PROPERTY}` AS id, n {{.*}} AS props",
            params={"ids": list(ids)},
        )
        documents_by_id = {}
        for row in rows:
            doc_id, props = row[0], dict(row[1])
            props.pop(DOC_ID_PROPERTY, None)
            text = props.pop(self.text_node_property, "")
            props.pop(self.embedding_node_property, None)
            documents_by_id[doc_id] = Document(
                id=doc_id, page_content=text, metadata=props
            )
        return [documents_by_id[id_] for id_ in ids if id_ in documents_by_id]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the vector store.

        Args:
            ids: The ids of the documents to delete. If ``None``, all
                documents under the store's node label are deleted.

        Returns:
            ``True`` on success.
        """
        if ids is None:
            self._query(f"MATCH (n:`{self.node_label}`) DELETE n")
            return True

        self._query(
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{DOC_ID_PROPERTY}` IN $ids "
            "DELETE n",
            params={"ids": list(ids)},
        )
        return True

    def update_documents(
        self,
        document_id: str,
        document: Document,
    ) -> None:
        """Update an existing document in the store by id.

        The new content is re-embedded so that similarity search reflects
        the update.

        Args:
            document_id: The id of the document to update.
            document: The new document content and metadata.
        """
        existing_document = self._query(
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{DOC_ID_PROPERTY}` = $document_id "
            "RETURN n",
            params={"document_id": document_id},
        )
        if not existing_document:
            raise ValueError(f"Document with id {document_id} not found in the store.")

        embedding = self.embedding.embed_documents([document.page_content])[0]
        update_query = (
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{DOC_ID_PROPERTY}` = $document_id "
            f"SET n.`{self.text_node_property}` = $new_content, "
            f"n.`{self.embedding_node_property}` = vecf32($embedding)"
        )
        params: Dict[str, Any] = {
            "document_id": document_id,
            "new_content": document.page_content,
            "embedding": embedding,
        }
        if document.metadata:
            update_query += ", n += $metadata"
            params["metadata"] = document.metadata
        self._query(update_query, params=params)

    @classmethod
    def from_texts(
        cls: Type[FalkorDBVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        distance_strategy: Optional[DistanceStrategy] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """Return a FalkorDBVector initialized from texts and embeddings."""
        embeddings = embedding.embed_documents(list(texts))

        if distance_strategy is None:
            distance_strategy = DEFAULT_DISTANCE_STRATEGY

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> FalkorDBVector:
        store = cls(
            embedding=embedding,
            search_type=search_type,
            **kwargs,
        )
        store._ensure_node_index()
        if search_type == SearchType.HYBRID:
            store._ensure_fts_index()

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        return store

    @classmethod
    def from_existing_index(
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        node_label: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """Get an instance of an existing FalkorDB vector index.

        This method returns the store without inserting any new embeddings.
        """
        store = cls(
            embedding=embedding,
            node_label=node_label,
            search_type=search_type,
            **kwargs,
        )

        embedding_dimension, index_type, entity_label, entity_property = (
            store.retrieve_existing_node_index()
        )

        if not index_type:
            raise ValueError(
                f"The specified vector index node label `{node_label}` does not exist. "
                "Make sure to check if you spelled the node label correctly"
            )

        if (
            embedding_dimension is not None
            and store.embedding_dimension != embedding_dimension
        ):
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            if not fts_node_label:
                raise ValueError(
                    "The specified keyword index name does not exist. "
                    "Make sure to check if you spelled it correctly"
                )
            if fts_node_label != store.node_label:
                raise ValueError(
                    "Vector and keyword index don't index the same node label"
                )

        store._index_verified = True
        return store

    @classmethod
    def from_existing_relationship_index(
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        relation_type: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """Get an instance of an existing FalkorDB relationship vector index.

        This method returns the store without inserting any new embeddings.
        """
        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector index"
            )

        store = cls(
            embedding=embedding,
            relation_type=relation_type,
            index_type=IndexType.RELATIONSHIP,
            **kwargs,
        )

        embedding_dimension, index_type, entity_label, entity_property = (
            store.retrieve_existing_relationship_index()
        )

        if not index_type:
            raise ValueError(
                "The specified vector index on the relationship "
                f"{relation_type} does not exist. "
                "Make sure to check if you spelled it correctly"
            )

        if (
            embedding_dimension is not None
            and store.embedding_dimension != embedding_dimension
        ):
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        store._index_verified = True
        return store

    @classmethod
    def from_existing_graph(
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        database: str,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> FalkorDBVector:
        """Initialize a FalkorDBVector from an existing graph.

        This method embeds the given text properties of the existing nodes,
        stores the embeddings on the nodes and returns a store that
        searches over them. It creates the required indexes if they don't
        exist yet.

        Args:
            embedding: The ``Embeddings`` model to use.
            database: The name of the existing graph.
            node_label: The label of the nodes to embed and search.
            embedding_node_property: The name of the property the
                embeddings are stored in.
            text_node_properties: The text properties of the nodes to embed.
            search_type: Similarity search type to use.
            retrieval_query: Optional custom retrieval query.

        Returns:
            An instance of FalkorDBVector backed by the existing graph.

        Example:
            .. code-block:: python

                vectorstore = FalkorDBVector.from_existing_graph(
                    embedding=my_embedding,
                    database="my_graph",
                    node_label="Document",
                    embedding_node_property="embedding",
                    text_node_properties=["title", "content"],
                )
        """
        if not database:
            raise ValueError("Parameter `database` must be given")
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )

        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, score, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null"
                + "".join([f", `{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata"
            )

        store = cls(
            database=database,
            embedding=embedding,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        store._ensure_node_index()
        if search_type == SearchType.HYBRID:
            store._ensure_fts_index(text_node_properties)

        # Populate embeddings of nodes that don't have one yet, in batches.
        while True:
            fetch_query = (
                f"MATCH (n:`{node_label}`) "
                f"WHERE n.`{embedding_node_property}` IS null "
                "AND any(k IN $props WHERE n[k] IS NOT null) "
                "RETURN id(n) AS id, reduce(str='', k IN $props |"
                " str + '\\n' + k + ': ' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            data = store._query(fetch_query, params={"props": text_node_properties})
            if not data:
                break
            text_embeddings = embedding.embed_documents([el[1] for el in data])

            params = {
                "data": [
                    {"id": el[0], "embedding": embedding_value}
                    for el, embedding_value in zip(data, text_embeddings)
                ]
            }
            store._query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE id(n) = row.id "
                f"SET n.`{embedding_node_property}` = vecf32(row.embedding) "
                "RETURN count(*)",
                params=params,
            )
            if len(data) < 1000:
                break
        return store

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """Construct a FalkorDBVector from raw texts and pre-generated
        embeddings.

        Example:
            .. code-block:: python

                from langchain_falkordb import FalkorDBVector
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = FalkorDBVector.from_embeddings(
                    text_embedding_pairs, embeddings
                )
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with FalkorDBVector.

        Args:
            query: Query text to search for.
            k: Number of results to return. Defaults to 4.
            params: Additional query parameters. Defaults to ``None``.
            filter: Dictionary of equality conditions to filter on metadata.
                Defaults to ``None``.

        Returns:
            The ``k`` documents most similar to the query.
        """
        embedding = self.embedding.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to an embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of documents to return. Defaults to 4.
            filter: Dictionary of equality conditions to filter on metadata.
                Defaults to ``None``.
            params: Additional query parameters. Defaults to ``None``.

        Returns:
            The ``k`` documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, params=params, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to the query, with scores.

        For vector search the score is the distance returned by the index
        (lower is more similar); for hybrid search it is a normalized
        similarity (higher is more similar).

        Args:
            query: Text to look up documents similar to.
            k: Number of documents to return. Defaults to 4.
            params: Additional query parameters. Defaults to ``None``.
            filter: Dictionary of equality conditions to filter on metadata.
                Defaults to ``None``.

        Returns:
            The ``k`` documents most similar to the query, with their
            scores.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search by vector and return scored documents.

        Args:
            embedding: The embedding vector to compare against.
            k: The number of documents to return. Defaults to 4.
            params: Additional query parameters. Defaults to ``None``.
            filter: Dictionary of equality conditions to filter on metadata.
                Defaults to ``None``.

        Returns:
            A list of ``(document, score)`` tuples, most similar first.
        """
        params = params or {}
        if filter:
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be used in combination with "
                    "a hybrid search approach"
                )
            if self._index_type == IndexType.RELATIONSHIP:
                raise ValueError(
                    "Metadata filtering is only supported for node vector indexes"
                )
            distance_function = (
                "vec.euclideanDistance"
                if self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE
                else "vec.cosineDistance"
            )
            filter_snippet, filter_params = construct_metadata_filter(filter)
            index_query = (
                f"MATCH (n:`{self.node_label}`) "
                f"WHERE n.`{self.embedding_node_property}` IS NOT NULL "
                f"AND {filter_snippet} "
                f"WITH n AS node, {distance_function}("
                f"n.`{self.embedding_node_property}`, vecf32($embedding)) AS score "
                "ORDER BY score ASC LIMIT $k "
            )
        else:
            index_query = _get_search_index_query(self.search_type, self._index_type)
            filter_params = {}

        entity = (
            "relationship" if self._index_type == IndexType.RELATIONSHIP else "node"
        )
        retrieval_query = self.retrieval_query or (
            f"RETURN {entity}.`{self.text_node_property}` AS text, score, "
            f"{entity} {{.*}} AS metadata"
        )

        read_query = index_query + retrieval_query
        parameters = {
            "entity_label": (
                self.relation_type
                if self._index_type == IndexType.RELATIONSHIP
                else self.node_label
            ),
            "entity_property": self.embedding_node_property,
            "k": k,
            "embedding": embedding,
            "query": kwargs.get("query", ""),
            **params,
            **filter_params,
        }

        # FalkorDB's vector index counts tombstones of updated/deleted
        # entries toward the requested k, so fewer than k live results can
        # come back. Over-fetch with escalation until k results arrive, the
        # result count stops growing, or the cap is reached. The filtered
        # path scans without the index and needs no escalation.
        fetch_k = k
        previous_count: Optional[int] = None
        while True:
            parameters["fetch_k"] = fetch_k
            try:
                results = self._query(read_query, params=parameters)
            except Exception as e:
                if _MISSING_INDEX_ERROR in str(e):
                    # The vector index does not exist yet, so the store
                    # cannot contain any matching documents.
                    return []
                raise
            if (
                filter
                or len(results) >= k
                or len(results) == previous_count
                or fetch_k >= _MAX_FETCH_K
            ):
                break
            previous_count = len(results)
            fetch_k = min(max(fetch_k * 4, 16), _MAX_FETCH_K)
        results = results[:k]

        if any(result[0] is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            raise ValueError(
                "Inspect the `retrieval_query` and ensure it doesn't "
                "return None for the `text` column"
            )

        return_embeddings = bool(kwargs.get("return_embeddings"))
        docs_and_scores: List[Tuple[Document, float]] = []
        for result in results:
            text = result[0]
            score = result[1]
            raw_metadata = dict(result[2]) if result[2] else {}
            doc_id = raw_metadata.pop(DOC_ID_PROPERTY, None)
            raw_metadata.pop(self.text_node_property, None)
            embedding_value = raw_metadata.pop(self.embedding_node_property, None)
            metadata = {
                key: value for key, value in raw_metadata.items() if value is not None
            }
            if return_embeddings:
                if embedding_value is None:
                    if not self.retrieval_query:
                        raise ValueError(
                            "Make sure that none of the "
                            f"`{self.embedding_node_property}` properties on "
                            f"nodes with label `{self.node_label}` are missing"
                        )
                    raise ValueError(
                        "Inspect the `retrieval_query` and ensure it returns "
                        "the embedding in the metadata"
                    )
                metadata["embedding"] = [float(value) for value in embedding_value]
            docs_and_scores.append(
                (Document(id=doc_id, page_content=text, metadata=metadata), score)
            )
        return docs_and_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to the query
        AND diversity among the selected documents.

        Args:
            query: Search query text.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch to pass to the MMR
                algorithm. Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results, with 0 corresponding to
                maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter: Dictionary of equality conditions to filter on metadata.
                Defaults to ``None``.

        Returns:
            The documents selected by maximal marginal relevance.
        """
        query_embedding = self.embedding.embed_query(query)

        got_docs = self.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
            filter=filter,
            **kwargs,
        )

        got_embeddings = [doc.metadata["embedding"] for doc, _ in got_docs]

        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc.metadata["embedding"]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on the distance
        strategy.

        The returned function maps the raw similarity score to a relevance
        score in [0, 1], where 1 means most relevant.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        if self.search_type == SearchType.HYBRID:
            # Hybrid scores are already normalized similarities.
            return lambda score: score
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return self._euclidean_relevance_score_fn
        raise ValueError(
            "No supported normalization function"
            f" for distance_strategy of {self._distance_strategy}."
            " Consider providing relevance_score_fn to FalkorDBVector."
        )
