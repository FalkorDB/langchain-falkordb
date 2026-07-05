"""LangChain integration for FalkorDB."""

from typing import TYPE_CHECKING, Any

from langchain_falkordb.chains import FalkorDBQAChain
from langchain_falkordb.graphs import FalkorDBGraph
from langchain_falkordb.message_history import FalkorDBChatMessageHistory
from langchain_falkordb.utils import DistanceStrategy
from langchain_falkordb.vectorstores import FalkorDBVector, IndexType, SearchType

if TYPE_CHECKING:
    from langchain_falkordb.checkpoint import FalkorDBSaver

__all__ = [
    "DistanceStrategy",
    "FalkorDBChatMessageHistory",
    "FalkorDBGraph",
    "FalkorDBQAChain",
    "FalkorDBSaver",
    "FalkorDBVector",
    "IndexType",
    "SearchType",
]


def __getattr__(name: str) -> Any:
    # FalkorDBSaver needs the optional langgraph-checkpoint dependency, so
    # it is imported lazily to keep the base package importable without it.
    if name == "FalkorDBSaver":
        from langchain_falkordb.checkpoint import FalkorDBSaver

        return FalkorDBSaver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
