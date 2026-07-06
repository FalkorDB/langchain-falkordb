"""Integration tests for the FalkorDB LangGraph checkpointer.

These tests require a running FalkorDB instance and the ``langgraph``
test dependencies (skipped automatically when not installed).
"""

import os
from typing import Generator

import pytest

pytest.importorskip("langgraph.checkpoint")

from langgraph.checkpoint.base import (  # noqa: E402
    Checkpoint,
    empty_checkpoint,
)

from langchain_falkordb.checkpoint import FalkorDBSaver  # noqa: E402

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", 6379))


def make_checkpoint(saver: FalkorDBSaver, values: dict) -> tuple[Checkpoint, dict]:
    """Build a checkpoint carrying `values` and its new_versions map."""
    checkpoint = empty_checkpoint()
    versions = {channel: saver.get_next_version(None, None) for channel in values}
    checkpoint["channel_values"] = dict(values)
    checkpoint["channel_versions"] = dict(versions)
    return checkpoint, versions


@pytest.fixture()
def saver() -> Generator[FalkorDBSaver, None, None]:
    checkpointer = FalkorDBSaver(host=host, port=port, database="checkpoint_test")
    try:
        yield checkpointer
    finally:
        try:
            checkpointer._graph.delete()
        except Exception:
            # Best-effort cleanup: the graph may never have been
            # created server-side, which is fine.
            pass


def test_put_and_get_roundtrip(saver: FalkorDBSaver) -> None:
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    checkpoint, versions = make_checkpoint(saver, {"count": 1, "notes": ["a"]})

    saved_config = saver.put(
        config, checkpoint, {"source": "input", "step": -1}, versions
    )
    assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]

    result = saver.get_tuple({"configurable": {"thread_id": "t1", "checkpoint_ns": ""}})
    assert result is not None
    assert result.checkpoint["id"] == checkpoint["id"]
    assert result.checkpoint["channel_values"] == {"count": 1, "notes": ["a"]}
    assert result.metadata["source"] == "input"
    assert result.metadata["step"] == -1
    assert result.parent_config is None


def test_get_tuple_returns_none_for_unknown_thread(saver: FalkorDBSaver) -> None:
    assert saver.get_tuple({"configurable": {"thread_id": "missing"}}) is None


def test_get_tuple_by_checkpoint_id_and_parent(saver: FalkorDBSaver) -> None:
    config = {"configurable": {"thread_id": "t2", "checkpoint_ns": ""}}
    first, first_versions = make_checkpoint(saver, {"count": 1})
    first_config = saver.put(
        config, first, {"source": "input", "step": -1}, first_versions
    )

    second, second_versions = make_checkpoint(saver, {"count": 2})
    saver.put(first_config, second, {"source": "loop", "step": 0}, second_versions)

    # Exact fetch of the first checkpoint by id
    old = saver.get_tuple(first_config)
    assert old is not None
    assert old.checkpoint["id"] == first["id"]
    assert old.checkpoint["channel_values"] == {"count": 1}

    # Latest checkpoint records the first as its parent
    latest = saver.get_tuple({"configurable": {"thread_id": "t2", "checkpoint_ns": ""}})
    assert latest is not None
    assert latest.checkpoint["id"] == second["id"]
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == first["id"]


def test_list_order_before_and_limit(saver: FalkorDBSaver) -> None:
    config = {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}}
    ids = []
    for i in range(3):
        checkpoint, versions = make_checkpoint(saver, {"count": i})
        config = saver.put(config, checkpoint, {"source": "loop", "step": i}, versions)
        ids.append(checkpoint["id"])

    listed = list(saver.list({"configurable": {"thread_id": "t3"}}))
    assert [t.checkpoint["id"] for t in listed] == list(reversed(ids))

    before_last = list(
        saver.list(
            {"configurable": {"thread_id": "t3"}},
            before={"configurable": {"checkpoint_id": ids[2]}},
        )
    )
    assert [t.checkpoint["id"] for t in before_last] == [ids[1], ids[0]]

    limited = list(saver.list({"configurable": {"thread_id": "t3"}}, limit=1))
    assert [t.checkpoint["id"] for t in limited] == [ids[2]]


def test_list_metadata_filter(saver: FalkorDBSaver) -> None:
    config = {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}
    first, first_versions = make_checkpoint(saver, {"count": 1})
    config = saver.put(config, first, {"source": "input", "step": -1}, first_versions)
    second, second_versions = make_checkpoint(saver, {"count": 2})
    saver.put(config, second, {"source": "loop", "step": 0}, second_versions)

    filtered = list(
        saver.list({"configurable": {"thread_id": "t4"}}, filter={"source": "input"})
    )
    assert len(filtered) == 1
    assert filtered[0].checkpoint["id"] == first["id"]


def test_put_writes_and_pending_writes(saver: FalkorDBSaver) -> None:
    config = {"configurable": {"thread_id": "t5", "checkpoint_ns": ""}}
    checkpoint, versions = make_checkpoint(saver, {"count": 1})
    saved_config = saver.put(
        config, checkpoint, {"source": "input", "step": -1}, versions
    )

    saver.put_writes(saved_config, [("count", 10), ("notes", "x")], task_id="task-1")
    # Re-writing the same (task, idx) must NOT overwrite the stored value.
    saver.put_writes(saved_config, [("count", 999)], task_id="task-1")

    result = saver.get_tuple(saved_config)
    assert result is not None
    assert ("task-1", "count", 10) in result.pending_writes
    assert ("task-1", "notes", "x") in result.pending_writes
    assert ("task-1", "count", 999) not in result.pending_writes


def test_namespaces_are_isolated(saver: FalkorDBSaver) -> None:
    base = {"configurable": {"thread_id": "t6", "checkpoint_ns": ""}}
    nested = {"configurable": {"thread_id": "t6", "checkpoint_ns": "child"}}
    outer, outer_versions = make_checkpoint(saver, {"count": 1})
    inner, inner_versions = make_checkpoint(saver, {"count": 100})
    saver.put(base, outer, {"source": "input", "step": -1}, outer_versions)
    saver.put(nested, inner, {"source": "input", "step": -1}, inner_versions)

    got_outer = saver.get_tuple(base)
    got_inner = saver.get_tuple(nested)
    assert got_outer is not None and got_inner is not None
    assert got_outer.checkpoint["channel_values"] == {"count": 1}
    assert got_inner.checkpoint["channel_values"] == {"count": 100}


def test_delete_thread(saver: FalkorDBSaver) -> None:
    config_one = {"configurable": {"thread_id": "t7", "checkpoint_ns": ""}}
    config_two = {"configurable": {"thread_id": "t8", "checkpoint_ns": ""}}
    checkpoint_one, versions_one = make_checkpoint(saver, {"count": 1})
    checkpoint_two, versions_two = make_checkpoint(saver, {"count": 2})
    saved_one = saver.put(
        config_one, checkpoint_one, {"source": "input", "step": -1}, versions_one
    )
    saver.put_writes(saved_one, [("count", 5)], task_id="task-1")
    saver.put(config_two, checkpoint_two, {"source": "input", "step": -1}, versions_two)

    saver.delete_thread("t7")
    assert saver.get_tuple(config_one) is None
    remaining = saver.get_tuple(config_two)
    assert remaining is not None
    assert remaining.checkpoint["channel_values"] == {"count": 2}


def test_langgraph_state_persists_across_invocations(saver: FalkorDBSaver) -> None:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        total: int

    def add_one(state: State) -> dict:
        return {"total": state["total"] + 1}

    builder = StateGraph(State)
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", END)
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "graph-thread"}}
    result = graph.invoke({"total": 0}, config)
    assert result["total"] == 1

    # State must be recoverable from the checkpointer.
    state = graph.get_state(config)
    assert state.values["total"] == 1

    result = graph.invoke({"total": state.values["total"]}, config)
    assert result["total"] == 2


async def test_langgraph_async_invocation(saver: FalkorDBSaver) -> None:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        total: int

    def add_one(state: State) -> dict:
        return {"total": state["total"] + 1}

    builder = StateGraph(State)
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", END)
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "async-graph-thread"}}
    result = await graph.ainvoke({"total": 41}, config)
    assert result["total"] == 42

    state = await graph.aget_state(config)
    assert state.values["total"] == 42
