"""LangGraph checkpoint saver storing checkpoints in FalkorDB."""

from __future__ import annotations

import asyncio
import base64
import os
import random
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.runnables import RunnableConfig

try:
    from langgraph.checkpoint.base import (
        WRITES_IDX_MAP,
        BaseCheckpointSaver,
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        SerializerProtocol,
        get_checkpoint_id,
    )
except ImportError as e:
    raise ImportError(
        "Could not import the langgraph-checkpoint python package. "
        "Please install it with `pip install langchain-falkordb[langgraph]`."
    ) from e

try:
    from langgraph.checkpoint.base import get_checkpoint_metadata
except ImportError:  # pragma: no cover - older langgraph-checkpoint versions

    def get_checkpoint_metadata(
        config: RunnableConfig, metadata: CheckpointMetadata
    ) -> CheckpointMetadata:
        return metadata


CHECKPOINT_LABEL = "Checkpoint"
BLOB_LABEL = "CheckpointBlob"
WRITE_LABEL = "CheckpointWrite"


def _encode(typed: Tuple[str, bytes]) -> Dict[str, str]:
    """Encode a (type, bytes) pair for storage as node properties."""
    return {"type": typed[0], "blob": base64.b64encode(typed[1]).decode("ascii")}


def _decode(type_: str, blob: str) -> Tuple[str, bytes]:
    """Decode node properties back into a (type, bytes) pair."""
    return type_, base64.b64decode(blob.encode("ascii"))


class FalkorDBSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpoint saver that stores checkpoints in FalkorDB.

    Checkpoints, channel-value blobs and pending writes are stored as nodes
    in a dedicated graph, so agent state survives restarts and can be
    shared between processes.

    Args:
        host: FalkorDB host. Defaults to ``"localhost"``.
        port: FalkorDB port. Defaults to ``6379``.
        username: Optionally provide your username. Falls back to the
            ``FALKORDB_USERNAME`` environment variable.
        password: Optionally provide your password. Falls back to the
            ``FALKORDB_PASSWORD`` environment variable.
        ssl: Whether the connection should use SSL/TLS encryption.
            Defaults to ``False``.
        database: Name of the graph the checkpoints are stored in.
            Defaults to ``"langgraph_checkpoints"``.
        serde: Optional serializer override.

    Example:
        .. code-block:: python

            from langgraph.graph import StateGraph
            from langchain_falkordb.checkpoint import FalkorDBSaver

            checkpointer = FalkorDBSaver(host="localhost", port=6379)
            graph = builder.compile(checkpointer=checkpointer)
            graph.invoke(
                {"total": 0}, {"configurable": {"thread_id": "thread-1"}}
            )
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
        database: str = "langgraph_checkpoints",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        try:
            import falkordb
        except ImportError as e:
            raise ImportError(
                "Could not import falkordb python package. "
                "Please install it with `pip install falkordb`."
            ) from e

        try:
            self._driver = falkordb.FalkorDB(
                host=host,
                port=port,
                username=username or os.environ.get("FALKORDB_USERNAME"),
                password=password or os.environ.get("FALKORDB_PASSWORD"),
                ssl=ssl,
            )
        except Exception as e:
            raise ValueError(
                "Could not connect to FalkorDB database. "
                "Please ensure that the host, port, username, "
                "and password are correct."
            ) from e
        self.database_name = database
        self._graph = self._driver.select_graph(database)
        for label in (CHECKPOINT_LABEL, BLOB_LABEL, WRITE_LABEL):
            try:
                self._graph.create_node_range_index(label, "thread_id")
            except Exception as e:
                if "already indexed" not in str(e).lower():
                    raise

    def _query(self, query: str, params: Optional[dict] = None) -> List[List[Any]]:
        return self._graph.query(query, params or {}).result_set

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[Tuple[str, str, Any]]:
        rows = self._query(
            f"MATCH (w:{WRITE_LABEL} {{thread_id: $thread_id, "
            "checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id}) "
            "RETURN w.task_id, w.channel, w.type, w.blob "
            "ORDER BY w.task_id, w.idx",
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        )
        return [
            (task_id, channel, self.serde.loads_typed(_decode(type_, blob)))
            for task_id, channel, type_, blob in rows
        ]

    def _load_channel_values(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> Dict[str, Any]:
        if not versions:
            return {}
        rows = self._query(
            "UNWIND $pairs AS pair "
            f"MATCH (b:{BLOB_LABEL} {{thread_id: $thread_id, "
            "checkpoint_ns: $checkpoint_ns, channel: pair.channel, "
            "version: pair.version}) "
            "RETURN b.channel, b.type, b.blob",
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "pairs": [
                    {"channel": channel, "version": version}
                    for channel, version in versions.items()
                ],
            },
        )
        return {
            channel: self.serde.loads_typed(_decode(type_, blob))
            for channel, type_, blob in rows
            if type_ != "empty"
        }

    def _tuple_from_row(
        self,
        config: Optional[RunnableConfig],
        thread_id: str,
        checkpoint_ns: str,
        row: List[Any],
    ) -> CheckpointTuple:
        checkpoint_id, type_, blob, metadata_type, metadata_blob, parent_id = row
        checkpoint: Checkpoint = self.serde.loads_typed(_decode(type_, blob))
        checkpoint["channel_values"] = self._load_channel_values(
            thread_id, checkpoint_ns, checkpoint["channel_versions"]
        )
        return CheckpointTuple(
            config=config
            or {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=self.serde.loads_typed(_decode(metadata_type, metadata_blob)),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=self._load_pending_writes(
                thread_id, checkpoint_ns, checkpoint_id
            ),
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from FalkorDB.

        If the config contains a ``checkpoint_id``, that exact checkpoint is
        retrieved; otherwise the latest checkpoint for the thread and
        namespace is returned.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            rows = self._query(
                f"MATCH (c:{CHECKPOINT_LABEL} {{thread_id: $thread_id, "
                "checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id}) "
                "RETURN c.checkpoint_id, c.type, c.blob, "
                "c.metadata_type, c.metadata_blob, c.parent_checkpoint_id",
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                },
            )
            if rows:
                return self._tuple_from_row(config, thread_id, checkpoint_ns, rows[0])
        else:
            rows = self._query(
                f"MATCH (c:{CHECKPOINT_LABEL} {{thread_id: $thread_id, "
                "checkpoint_ns: $checkpoint_ns}) "
                "RETURN c.checkpoint_id, c.type, c.blob, "
                "c.metadata_type, c.metadata_blob, c.parent_checkpoint_id "
                "ORDER BY c.checkpoint_id DESC LIMIT 1",
                {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
            )
            if rows:
                return self._tuple_from_row(None, thread_id, checkpoint_ns, rows[0])
        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from FalkorDB, newest first.

        Args:
            config: Base configuration (thread, namespace, checkpoint id)
                for filtering checkpoints.
            filter: Metadata key/value pairs that must all match.
            before: Only yield checkpoints created before this one.
            limit: Maximum number of checkpoints to yield.
        """
        clauses = []
        params: Dict[str, Any] = {}
        if config is not None:
            clauses.append("c.thread_id = $thread_id")
            params["thread_id"] = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                clauses.append("c.checkpoint_ns = $checkpoint_ns")
                params["checkpoint_ns"] = checkpoint_ns
            if checkpoint_id := get_checkpoint_id(config):
                clauses.append("c.checkpoint_id = $config_checkpoint_id")
                params["config_checkpoint_id"] = checkpoint_id
        if before and (before_id := get_checkpoint_id(before)):
            clauses.append("c.checkpoint_id < $before_checkpoint_id")
            params["before_checkpoint_id"] = before_id

        where = ("WHERE " + " AND ".join(clauses) + " ") if clauses else ""
        rows = self._query(
            f"MATCH (c:{CHECKPOINT_LABEL}) "
            + where
            + "RETURN c.thread_id, c.checkpoint_ns, c.checkpoint_id, c.type, "
            "c.blob, c.metadata_type, c.metadata_blob, c.parent_checkpoint_id "
            "ORDER BY c.checkpoint_id DESC",
            params,
        )

        remaining = limit
        for row in rows:
            if remaining is not None and remaining <= 0:
                break
            thread_id, checkpoint_ns = row[0], row[1]
            metadata = self.serde.loads_typed(_decode(row[5], row[6]))
            if filter and not all(
                metadata.get(key) == value for key, value in filter.items()
            ):
                continue
            if remaining is not None:
                remaining -= 1
            yield self._tuple_from_row(None, thread_id, checkpoint_ns, row[2:])

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint (and its new channel values) to FalkorDB."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        stored: Dict[str, Any] = dict(checkpoint)
        values: Dict[str, Any] = stored.pop("channel_values", {})

        if new_versions:
            blob_rows = []
            for channel, version in new_versions.items():
                typed = (
                    self.serde.dumps_typed(values[channel])
                    if channel in values
                    else ("empty", b"")
                )
                blob_rows.append(
                    {"channel": channel, "version": version, **_encode(typed)}
                )
            self._query(
                "UNWIND $blobs AS row "
                f"MERGE (b:{BLOB_LABEL} {{thread_id: $thread_id, "
                "checkpoint_ns: $checkpoint_ns, channel: row.channel, "
                "version: row.version}) "
                "SET b.type = row.type, b.blob = row.blob",
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "blobs": blob_rows,
                },
            )

        checkpoint_encoded = _encode(self.serde.dumps_typed(stored))
        metadata_encoded = _encode(
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        )
        self._query(
            f"MERGE (c:{CHECKPOINT_LABEL} {{thread_id: $thread_id, "
            "checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id}) "
            "SET c.type = $type, c.blob = $blob, "
            "c.metadata_type = $metadata_type, c.metadata_blob = $metadata_blob, "
            "c.parent_checkpoint_id = $parent_checkpoint_id",
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
                "type": checkpoint_encoded["type"],
                "blob": checkpoint_encoded["blob"],
                "metadata_type": metadata_encoded["type"],
                "metadata_blob": metadata_encoded["blob"],
                "parent_checkpoint_id": config["configurable"].get("checkpoint_id")
                or "",
            },
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save a list of pending writes for a checkpoint.

        Writes on regular channels are only stored once per (task, index);
        writes on special channels (see ``WRITES_IDX_MAP``) are upserted.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        keep_rows: List[Dict[str, Any]] = []
        upsert_rows: List[Dict[str, Any]] = []
        for idx, (channel, value) in enumerate(writes):
            final_idx = WRITES_IDX_MAP.get(channel, idx)
            row = {
                "task_id": task_id,
                "idx": final_idx,
                "channel": channel,
                "task_path": task_path,
                **_encode(self.serde.dumps_typed(value)),
            }
            (upsert_rows if final_idx < 0 else keep_rows).append(row)

        params = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        if keep_rows:
            self._query(
                "UNWIND $rows AS row "
                f"MERGE (w:{WRITE_LABEL} {{thread_id: $thread_id, "
                "checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id, "
                "task_id: row.task_id, idx: row.idx}) "
                "ON CREATE SET w.channel = row.channel, w.type = row.type, "
                "w.blob = row.blob, w.task_path = row.task_path",
                {**params, "rows": keep_rows},
            )
        if upsert_rows:
            self._query(
                "UNWIND $rows AS row "
                f"MERGE (w:{WRITE_LABEL} {{thread_id: $thread_id, "
                "checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id, "
                "task_id: row.task_id, idx: row.idx}) "
                "SET w.channel = row.channel, w.type = row.type, "
                "w.blob = row.blob, w.task_path = row.task_path",
                {**params, "rows": upsert_rows},
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, blobs and writes for a thread."""
        for label in (CHECKPOINT_LABEL, BLOB_LABEL, WRITE_LABEL):
            self._query(
                f"MATCH (n:{label} {{thread_id: $thread_id}}) DELETE n",
                {"thread_id": thread_id},
            )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronous version of :meth:`get_tuple`."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous version of :meth:`list`."""
        results = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in results:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronous version of :meth:`put`."""
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronous version of :meth:`put_writes`."""
        return await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronous version of :meth:`delete_thread`."""
        return await asyncio.to_thread(self.delete_thread, thread_id)

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        """Generate the next monotonically increasing channel version."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
