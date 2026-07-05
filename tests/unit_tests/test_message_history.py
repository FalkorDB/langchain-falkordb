"""Unit tests for FalkorDBChatMessageHistory (no database required)."""

import pytest

from langchain_falkordb.message_history import FalkorDBChatMessageHistory


def test_empty_session_id_raises() -> None:
    with pytest.raises(ValueError, match="session_id"):
        FalkorDBChatMessageHistory("")
