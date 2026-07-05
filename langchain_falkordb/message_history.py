"""FalkorDB chat message history integration for LangChain."""

import logging
import os
from typing import Any, List, Optional, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)

_SUPPORTED_MESSAGE_TYPES = ("human", "ai", "system")


class FalkorDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a FalkorDB database.

    This class handles storing and retrieving chat messages in a FalkorDB
    database. It creates a session and stores messages in a message chain,
    maintaining a link between subsequent messages.

    Args:
        session_id: The session ID for storing and retrieving messages,
            also the name of the graph the messages are stored in.
        username: Username for authenticating with FalkorDB. Falls back to
            the ``FALKORDB_USERNAME`` environment variable.
        password: Password for authenticating with FalkorDB. Falls back to
            the ``FALKORDB_PASSWORD`` environment variable.
        host: Host where FalkorDB is running. Defaults to ``"localhost"``.
        port: Port number where FalkorDB is running. Defaults to ``6379``.
        node_label: Label for the session node in the graph.
            Defaults to ``"Session"``.
        window: The number of message pairs to retrieve when querying the
            history. Defaults to 3.
        ssl: Whether to use SSL for connecting to the database.
            Defaults to ``False``.
        graph: Optional graph object to reuse an existing connection. Any
            object exposing a ``_driver`` attribute holding a
            ``falkordb.FalkorDB`` client is accepted, e.g. a
            ``FalkorDBGraph`` instance from ``langchain-community``.

    Example:
        .. code-block:: python

            from langchain_core.messages import HumanMessage
            from langchain_falkordb import FalkorDBChatMessageHistory

            history = FalkorDBChatMessageHistory(
                session_id="1234",
                host="localhost",
                port=6379,
            )
            history.add_message(HumanMessage(content="Hello!"))
    """

    def __init__(
        self,
        session_id: Union[str, int],
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        node_label: str = "Session",
        window: int = 3,
        ssl: bool = False,
        *,
        graph: Optional[Any] = None,
    ) -> None:
        try:
            import falkordb
        except ImportError as e:
            raise ImportError(
                "Could not import falkordb python package. "
                "Please install it with `pip install falkordb`."
            ) from e

        if not session_id:
            raise ValueError("Please ensure that the session_id parameter is provided.")
        # node_label is interpolated into Cypher inside backtick-quoted
        # identifiers; an embedded backtick would escape the identifier.
        if "`" in node_label:
            raise ValueError("`node_label` must not contain backtick characters")

        if graph is not None:
            # Reuse the connection of a FalkorDBGraph-like object.
            self._driver = graph._driver
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
                    "Please ensure that the host, port, "
                    "username, and password are correct."
                ) from e

        self._database = self._driver.select_graph(str(session_id))
        self._session_id = session_id
        self._node_label = node_label
        self._window = window

        self._database.query(
            f"MERGE (s:`{self._node_label}` {{id: $session_id}})",
            {"session_id": self._session_id},
        )

        try:
            self._database.create_node_range_index(self._node_label, "id")
        except Exception as e:
            # Re-connecting to an existing session is expected; the index
            # then already exists.
            if "already indexed" not in str(e).lower():
                raise

    def _process_records(self, records: list) -> List[BaseMessage]:
        """Convert FalkorDB records into ``BaseMessage`` objects.

        Args:
            records: The raw records fetched from the FalkorDB query.

        Returns:
            The corresponding list of ``BaseMessage`` objects.
        """
        messages: List[BaseMessage] = []

        for record in records:
            content = record[0].get("data", {}).get("content", "")
            message_type = record[0].get("type", "").lower()

            if message_type == "human":
                messages.append(HumanMessage(content=content))
            elif message_type == "ai":
                messages.append(AIMessage(content=content))
            elif message_type == "system":
                messages.append(SystemMessage(content=content))
            else:
                raise ValueError(f"Unknown message type: {message_type}")

        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from FalkorDB for the session.

        Returns:
            The messages in the current session.
        """
        query = (
            f"MATCH (s:`{self._node_label}`)-[:LAST_MESSAGE]->(last_message) "
            "MATCH p=(last_message)<-[:NEXT*0.."
            f"{self._window * 2}]-() WITH p, length(p) AS length "
            "ORDER BY length DESC LIMIT 1 UNWIND reverse(nodes(p)) AS node "
            "RETURN {data:{content: node.content}, type:node.type} AS result"
        )

        records = self._database.query(query).result_set

        return self._process_records(records)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        """Block direct assignment to ``messages`` to prevent misuse."""
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_message' method instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the session in FalkorDB.

        Args:
            message: The message object to add to the session. Must be a
                human, AI, or system message.
        """
        if message.type not in _SUPPORTED_MESSAGE_TYPES:
            raise ValueError(
                f"Unsupported message type: {message.type!r}. Supported "
                f"message types are: {', '.join(_SUPPORTED_MESSAGE_TYPES)}"
            )
        create_query = (
            f"MATCH (s:`{self._node_label}`) "
            "CREATE (new:Message {type: $type, content: $content}) "
            "WITH s, new "
            "OPTIONAL MATCH (s)-[lm:LAST_MESSAGE]->(last_message:Message) "
            "FOREACH (_ IN CASE WHEN last_message IS NULL THEN [] ELSE [1] END | "
            "  MERGE (last_message)-[:NEXT]->(new)) "
            "MERGE (s)-[:LAST_MESSAGE]->(new) "
        )

        self._database.query(
            create_query,
            {
                "type": message.type,
                "content": message.content,
            },
        )

    def clear(self) -> None:
        """Clear all messages from the session in FalkorDB.

        Deletes all messages linked to the session and resets the message
        history.
        """
        query = (
            f"MATCH (s:`{self._node_label}`)-[:LAST_MESSAGE|NEXT*0..]->(m:Message) "
            "WITH m DELETE m"
        )
        self._database.query(query)
