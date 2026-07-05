"""Test that the public API is importable from the package root."""

EXPECTED_ALL = {
    "DistanceStrategy",
    "FalkorDBChatMessageHistory",
    "FalkorDBVector",
    "IndexType",
    "SearchType",
}


def test_all_imports() -> None:
    import langchain_falkordb

    assert set(langchain_falkordb.__all__) == EXPECTED_ALL


def test_public_symbols_importable() -> None:
    from langchain_falkordb import (
        DistanceStrategy,
        FalkorDBChatMessageHistory,
        FalkorDBVector,
        IndexType,
        SearchType,
    )

    assert FalkorDBVector is not None
    assert FalkorDBChatMessageHistory is not None
    assert SearchType is not None
    assert IndexType is not None
    assert DistanceStrategy is not None
