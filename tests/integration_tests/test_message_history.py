"""Integration tests for FalkorDB chat message history.

These tests require a running FalkorDB instance. By default they connect to
``localhost:6379``; set ``FALKORDB_HOST`` / ``FALKORDB_PORT`` to override.
"""

import os

import pytest
from falkordb import FalkorDB
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_falkordb.message_history import FalkorDBChatMessageHistory

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", 6379))


def drop_session_graph(history: FalkorDBChatMessageHistory) -> None:
    try:
        history._database.delete()
    except Exception:
        pass


def test_add_messages() -> None:
    """Basic testing: add messages to the FalkorDBChatMessageHistory."""
    message_store = FalkorDBChatMessageHistory("500daysofSadiya", host=host, port=port)
    message_store_another = FalkorDBChatMessageHistory(
        "Shebrokemyheart", host=host, port=port
    )
    try:
        message_store.clear()
        assert len(message_store.messages) == 0
        message_store.add_user_message("Hello! Language Chain!")
        message_store.add_ai_message("Hi Guys!")

        # Create another message store to
        # check if the messages are stored correctly.
        message_store_another.clear()
        assert len(message_store_another.messages) == 0
        message_store_another.add_user_message("Hello! Bot!")
        message_store_another.add_ai_message("Hi there!")
        message_store_another.add_user_message("How's this pr going?")

        # Now check if the messages are stored in the database correctly.
        assert len(message_store.messages) == 2
        assert isinstance(message_store.messages[0], HumanMessage)
        assert isinstance(message_store.messages[1], AIMessage)
        assert message_store.messages[0].content == "Hello! Language Chain!"
        assert message_store.messages[1].content == "Hi Guys!"

        assert len(message_store_another.messages) == 3
        assert isinstance(message_store_another.messages[0], HumanMessage)
        assert isinstance(message_store_another.messages[1], AIMessage)
        assert isinstance(message_store_another.messages[2], HumanMessage)
        assert message_store_another.messages[0].content == "Hello! Bot!"
        assert message_store_another.messages[1].content == "Hi there!"
        assert message_store_another.messages[2].content == "How's this pr going?"

        # Now clear the first history.
        message_store.clear()
        assert len(message_store.messages) == 0
        assert len(message_store_another.messages) == 3
        message_store_another.clear()
        assert len(message_store.messages) == 0
        assert len(message_store_another.messages) == 0
    finally:
        drop_session_graph(message_store)
        drop_session_graph(message_store_another)


def test_reconnect_to_existing_session() -> None:
    """Re-instantiating a history for an existing session must not raise."""
    session_id = "reconnect_session_test"
    first = FalkorDBChatMessageHistory(session_id, host=host, port=port)
    try:
        first.clear()
        first.add_user_message("Hello!")
        first.add_ai_message("Hi! How can I help?")

        # Reconnecting to the same session used to raise
        # "Session has already been indexed".
        second = FalkorDBChatMessageHistory(session_id, host=host, port=port)
        messages = second.messages
        assert len(messages) == 2
        assert messages[0].content == "Hello!"
        assert messages[1].content == "Hi! How can I help?"
    finally:
        drop_session_graph(first)


def test_system_message_roundtrip() -> None:
    """System messages must be stored and read back correctly."""
    history = FalkorDBChatMessageHistory("system_msg_test", host=host, port=port)
    try:
        history.clear()
        history.add_message(SystemMessage(content="You are a helpful assistant"))
        history.add_user_message("Hello!")
        messages = history.messages
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are a helpful assistant"
        assert isinstance(messages[1], HumanMessage)
    finally:
        drop_session_graph(history)


def test_unsupported_message_type_raises_on_add() -> None:
    """Unsupported message types must fail on write, not corrupt the session."""
    history = FalkorDBChatMessageHistory("unsupported_msg_test", host=host, port=port)
    try:
        history.clear()
        with pytest.raises(ValueError, match="message type"):
            history.add_message(FunctionMessage(name="f", content="x"))
        # The session must still be readable.
        assert history.messages == []
    finally:
        drop_session_graph(history)


def test_add_messages_graph_object() -> None:
    """Basic testing: Passing driver through a graph-like object."""
    driver = FalkorDB(host=host, port=port)

    class DuckGraph:
        """Duck-typed stand-in for langchain FalkorDBGraph."""

        def __init__(self) -> None:
            self._driver = driver
            self._graph = driver.select_graph("NeverGonnaLetYouDown")

    message_store = FalkorDBChatMessageHistory(
        "Gonnahavetoteachmehowtoloveyouagain", graph=DuckGraph()
    )
    try:
        message_store.clear()
        assert len(message_store.messages) == 0
        message_store.add_user_message("Hello! Language Chain!")
        message_store.add_ai_message("Hi Guys!")
        # Now check if the messages are stored in the database correctly.
        assert len(message_store.messages) == 2
    finally:
        drop_session_graph(message_store)
