"""
Append-only Event Log — the source of truth for all raw agent interactions.

Design principles:
  - Writes are always <1ms (no LLM involved)
  - Never mutated or deleted — only appended
  - zstd-compressed for storage efficiency
  - Tracks consolidation offsets so the worker knows what's been processed
  - Thread-safe via SQLite WAL mode (concurrent readers, single writer)
"""

import json
import sqlite3
import zlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


@dataclass
class RawTurn:
    role: str                    # "user" | "assistant" | "tool"
    content: str
    session_id: str
    user_id: str
    turn_index: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)  # tool name, tool output, etc.


CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    content_gz      BLOB    NOT NULL,   -- zlib-compressed content
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    timestamp       TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_user    ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, turn_index);

-- Tracks which events have been consolidated by the MCM worker
CREATE TABLE IF NOT EXISTS consolidation_offsets (
    user_id         TEXT    PRIMARY KEY,
    last_event_id   INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


class EventLog:
    """
    Append-only SQLite event log with WAL mode for high concurrency.

    Multiple agent instances can write simultaneously.
    A single MCM consolidation worker reads from this log.
    """

    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append(self, turn: RawTurn) -> int:
        """
        Append a single turn to the log. Returns the assigned event ID.
        This is the only write path — it is fast (<1ms), no LLM involved.
        """
        content_gz = zlib.compress(turn.content.encode("utf-8"), level=6)
        metadata_json = json.dumps(turn.metadata)

        # Enforce metadata size limit (prevent silent truncation issues)
        if len(metadata_json) > 4096:
            metadata_json = json.dumps({"truncated": True, "original_keys": list(turn.metadata.keys())})

        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (user_id, session_id, turn_index, role, content_gz, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.user_id,
                    turn.session_id,
                    turn.turn_index,
                    turn.role,
                    content_gz,
                    metadata_json,
                    turn.timestamp,
                ),
            )
            return cursor.lastrowid

    def get_unconsolidated(self, user_id: str, batch_size: int = 100) -> tuple[list[RawTurn], int]:
        """
        Fetch turns that have not yet been consolidated for a given user.
        Returns (turns, last_event_id) — pass last_event_id to mark_consolidated().
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_event_id FROM consolidation_offsets WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            last_consolidated = row["last_event_id"] if row else 0

            rows = conn.execute(
                """
                SELECT id, user_id, session_id, turn_index, role, content_gz, metadata_json, timestamp
                FROM events
                WHERE user_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (user_id, last_consolidated, batch_size),
            ).fetchall()

        if not rows:
            return [], last_consolidated

        turns = []
        for r in rows:
            content = zlib.decompress(r["content_gz"]).decode("utf-8")
            turns.append(
                RawTurn(
                    role=r["role"],
                    content=content,
                    session_id=r["session_id"],
                    user_id=r["user_id"],
                    turn_index=r["turn_index"],
                    timestamp=r["timestamp"],
                    metadata=json.loads(r["metadata_json"]),
                )
            )

        last_event_id = rows[-1]["id"]
        return turns, last_event_id

    def mark_consolidated(self, user_id: str, last_event_id: int) -> None:
        """Mark all events up to last_event_id as consolidated for this user."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO consolidation_offsets (user_id, last_event_id, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET
                    last_event_id = excluded.last_event_id,
                    updated_at = excluded.updated_at
                """,
                (user_id, last_event_id),
            )

    def get_session_log(self, user_id: str, session_id: str) -> list[RawTurn]:
        """Retrieve the full raw log for a session (for full-context fallback)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content_gz, metadata_json, session_id, user_id, turn_index, timestamp
                FROM events
                WHERE user_id = ? AND session_id = ?
                ORDER BY turn_index ASC
                """,
                (user_id, session_id),
            ).fetchall()

        return [
            RawTurn(
                role=r["role"],
                content=zlib.decompress(r["content_gz"]).decode("utf-8"),
                session_id=r["session_id"],
                user_id=r["user_id"],
                turn_index=r["turn_index"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    def unconsolidated_count(self, user_id: str) -> int:
        """How many turns are pending consolidation for a user."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_event_id FROM consolidation_offsets WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            last_consolidated = row["last_event_id"] if row else 0
            count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE user_id = ? AND id > ?",
                (user_id, last_consolidated),
            ).fetchone()[0]
        return count
