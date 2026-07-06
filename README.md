# langchain-falkordb

[![PyPI version](https://img.shields.io/pypi/v/langchain-falkordb.svg)](https://pypi.org/project/langchain-falkordb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/FalkorDB/langchain-falkordb/actions/workflows/ci.yml/badge.svg)](https://github.com/FalkorDB/langchain-falkordb/actions/workflows/ci.yml)

This package contains the [LangChain](https://github.com/langchain-ai/langchain)
integration for [FalkorDB](https://www.falkordb.com/), a low-latency graph
database with native vector and full-text indexing.

It provides:

- **`FalkorDBGraph`** — a graph wrapper with schema introspection and
  `GraphDocument` ingestion, for building knowledge graphs.
- **`FalkorDBVector`** — a LangChain
  [vector store](https://python.langchain.com/docs/concepts/vectorstores/)
  backed by FalkorDB vector indexes, with support for metadata filtering,
  maximal marginal relevance (MMR) search, and hybrid (vector + full-text)
  search.
- **`FalkorDBQAChain`** — a natural-language-to-Cypher question-answering
  chain over a FalkorDB graph.
- **`FalkorDBSaver`** — a [LangGraph](https://github.com/langchain-ai/langgraph)
  checkpointer that persists agent state in FalkorDB
  (`pip install langchain-falkordb[langgraph]`).
- **`FalkorDBChatMessageHistory`** — a LangChain
  [chat message history](https://python.langchain.com/docs/concepts/chat_history/)
  that persists conversations in FalkorDB.

## Installation

```bash
pip install langchain-falkordb
```

You also need a running FalkorDB instance. The easiest way is Docker:

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb:latest
```

or use a free [FalkorDB Cloud](https://app.falkordb.cloud/) instance.

## Graph wrapper

`FalkorDBGraph` gives you direct Cypher access with schema introspection,
and ingests `GraphDocument` objects (e.g. produced by an LLM graph
transformer):

```python
from langchain_core.documents import Document
from langchain_falkordb import FalkorDBGraph
from langchain_falkordb.graphs import GraphDocument, Node, Relationship

graph = FalkorDBGraph("movies", host="localhost", port=6379)

tom = Node(id="Tom Hanks", type="Actor")
gump = Node(id="Forrest Gump", type="Movie")
graph.add_graph_documents(
    [
        GraphDocument(
            nodes=[tom, gump],
            relationships=[Relationship(source=tom, target=gump, type="ACTED_IN")],
            source=Document(page_content="Tom Hanks acted in Forrest Gump."),
        )
    ],
    include_source=True,  # links entities to their source Document node
)

graph.refresh_schema()
print(graph.get_schema)
print(graph.query("MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.id, m.id"))
```

It also works with `LLMGraphTransformer` from `langchain-experimental` to
build knowledge graphs from unstructured text: pass the transformer's
`GraphDocument` output straight to `add_graph_documents`.

## Vector store

`FalkorDBVector` works with any
[`Embeddings`](https://python.langchain.com/docs/concepts/embedding_models/)
implementation. The examples below use `OpenAIEmbeddings` from
[`langchain-openai`](https://pypi.org/project/langchain-openai/).

```python
from langchain_falkordb import FalkorDBVector
from langchain_openai import OpenAIEmbeddings

vectorstore = FalkorDBVector.from_texts(
    texts=[
        "FalkorDB is a graph database",
        "LangChain is a framework for LLM applications",
    ],
    embedding=OpenAIEmbeddings(),
    host="localhost",
    port=6379,
    database="my_knowledge_base",  # optional; generated if omitted
)

results = vectorstore.similarity_search("What is FalkorDB?", k=1)
print(results[0].page_content)
```

### Adding and managing documents

```python
from langchain_core.documents import Document

vectorstore.add_documents(
    [Document(page_content="FalkorDB supports vector search", metadata={"topic": "search"})],
    ids=["doc-1"],
)

vectorstore.get_by_ids(["doc-1"])  # fetch by id
vectorstore.delete(["doc-1"])      # delete by id
```

Adding a document with an existing id overwrites it (upsert semantics).

### Metadata filtering

```python
results = vectorstore.similarity_search(
    "graph databases",
    k=4,
    filter={"topic": "search"},
)
```

Filter values are always passed as query parameters, never interpolated
into the Cypher query.

### Maximal marginal relevance (MMR) search

```python
results = vectorstore.max_marginal_relevance_search(
    "graph databases", k=4, fetch_k=20, lambda_mult=0.5
)
```

### Hybrid search

Hybrid search combines the vector index with a full-text index over the
document text:

```python
from langchain_falkordb import FalkorDBVector, SearchType

vectorstore = FalkorDBVector.from_texts(
    texts=["FalkorDB is a graph database"],
    embedding=OpenAIEmbeddings(),
    search_type=SearchType.HYBRID,
)
```

### Reusing existing indexes and graphs

```python
# Connect to a vector index that already contains data
store = FalkorDBVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    database="my_knowledge_base",
    node_label="Chunk",
)

# Embed and search text properties of an existing graph
store = FalkorDBVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    database="my_graph",
    node_label="Document",
    embedding_node_property="embedding",
    text_node_properties=["title", "content"],
)
```

## Question answering over a graph

`FalkorDBQAChain` turns a natural-language question into Cypher, runs it,
and phrases the answer:

```python
from langchain_falkordb import FalkorDBGraph, FalkorDBQAChain
from langchain_openai import ChatOpenAI

graph = FalkorDBGraph("movies")
chain = FalkorDBQAChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini"),
    graph=graph,
    allow_dangerous_requests=True,  # explicit opt-in, see security note below
)
print(chain.invoke({"query": "Who acted in Forrest Gump?"})["result"])
```

> **Security note**: the chain executes LLM-generated Cypher against your
> database. Use narrowly-scoped credentials and set
> `allow_dangerous_requests=True` only after understanding the risks.

## LangGraph checkpointer

`FalkorDBSaver` persists LangGraph agent state in FalkorDB, so threads
survive restarts and can be shared between processes:

```bash
pip install langchain-falkordb[langgraph]
```

```python
from langchain_falkordb.checkpoint import FalkorDBSaver

checkpointer = FalkorDBSaver(host="localhost", port=6379)
graph = builder.compile(checkpointer=checkpointer)  # any StateGraph builder
graph.invoke({"total": 0}, {"configurable": {"thread_id": "thread-1"}})
```

## Chat message history

```python
from langchain_falkordb import FalkorDBChatMessageHistory

history = FalkorDBChatMessageHistory(
    session_id="user-42",
    host="localhost",
    port=6379,
)

history.add_user_message("Hello!")
history.add_ai_message("Hi! How can I help?")
print(history.messages)
```

Each session is stored in its own graph named after the ``session_id``, so
histories are isolated per session and survive reconnects.

## Authentication

For protected instances (e.g. FalkorDB Cloud), pass ``username`` /
``password`` (and ``ssl=True`` if applicable) to the constructors, or set
the ``FALKORDB_USERNAME`` and ``FALKORDB_PASSWORD`` environment variables.

## Development

```bash
git clone https://github.com/FalkorDB/langchain-falkordb.git
cd langchain-falkordb
poetry install --with test,lint,typing
```

Run unit tests (no services needed):

```bash
poetry run pytest tests/unit_tests --disable-socket
```

Run integration tests (requires FalkorDB on ``localhost:6379``, override
with ``FALKORDB_HOST`` / ``FALKORDB_PORT``):

```bash
docker run -d -p 6379:6379 falkordb/falkordb:latest
poetry run pytest tests/integration_tests
```

Lint and type-check:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy langchain_falkordb
```

## License

[MIT](LICENSE)
