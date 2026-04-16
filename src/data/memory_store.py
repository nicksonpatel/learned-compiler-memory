"""
Versioned Memory Store — immutable, append-only memory units with supersession.

Design principles to fix mem0's staleness problem:
  - Memory units are NEVER mutated in place
  - Contradictions are resolved by creating a new unit that supersedes old ones
  - Retrieval always filters to the latest non-superseded version
  - Full history is retained for audit / rollback
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_units (
    id              TEXT    PRIMARY KEY,          -- uuid
    user_id         TEXT    NOT NULL,
    type            TEXT    NOT NULL,             -- fact | preference | task_summary | abstraction | open_thread
    content         TEXT    NOT NULL,
    domain          TEXT,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    tags_json       TEXT    NOT NULL DEFAULT '[]',
    source_turns    TEXT    NOT NULL DEFAULT '[]', -- JSON list of event log IDs
    version         INTEGER NOT NULL DEFAULT 1,
    supersedes_json TEXT    NOT NULL DEFAULT '[]', -- JSON list of unit IDs this replaces
    is_superseded   INTEGER NOT NULL DEFAULT 0,    -- 0=active, 1=superseded
    valid_from      TEXT    NOT NULL,
    valid_until     TEXT,                          -- NULL = currently valid
    embedding_model TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_units_user        ON memory_units(user_id, is_superseded);
CREATE INDEX IF NOT EXISTS idx_units_type        ON memory_units(user_id, type, is_superseded);
CREATE INDEX IF NOT EXISTS idx_units_domain      ON memory_units(user_id, domain, is_superseded);
CREATE INDEX IF NOT EXISTS idx_units_confidence  ON memory_units(user_id, confidence DESC);
"""


class MemoryStore:
    """
    Immutable, versioned memory unit store.

    On contradiction: old unit is marked superseded, new unit references it.
    This eliminates silent staleness — you can always see what changed and why.
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

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

    def write_unit(
        self,
        user_id: str,
        unit_type: str,
        content: str,
        domain: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        source_turns: list[int] | None = None,
        supersedes: list[str] | None = None,
    ) -> str:
        """
        Write a new memory unit. If supersedes is non-empty, marks those units as superseded.
        Returns the new unit's ID.
        """
        unit_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            # Mark old units as superseded
            if supersedes:
                for old_id in supersedes:
                    conn.execute(
                        "UPDATE memory_units SET is_superseded = 1, valid_until = ? WHERE id = ? AND user_id = ?",
                        (now, old_id, user_id),
                    )

            conn.execute(
                """
                INSERT INTO memory_units
                  (id, user_id, type, content, domain, confidence, tags_json,
                   source_turns, version, supersedes_json, is_superseded, valid_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0, ?)
                """,
                (
                    unit_id,
                    user_id,
                    unit_type,
                    content,
                    domain,
                    min(max(confidence, 0.0), 1.0),  # clamp to [0, 1]
                    json.dumps(tags or []),
                    json.dumps(source_turns or []),
                    json.dumps(supersedes or []),
                    now,
                ),
            )

        return unit_id

    def get_active_units(
        self,
        user_id: str,
        unit_type: str | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Fetch all active (non-superseded) memory units for a user.
        Supports optional filtering by type, domain, and minimum confidence.
        """
        query = "SELECT * FROM memory_units WHERE user_id = ? AND is_superseded = 0"
        params: list[Any] = [user_id]

        if unit_type:
            query += " AND type = ?"
            params.append(unit_type)
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        query += " ORDER BY confidence DESC, valid_from DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_unit_by_id(self, unit_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE id = ?", (unit_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_history(self, user_id: str, unit_id: str) -> list[dict[str, Any]]:
        """
        Walk the supersession chain to see the full history of a memory unit.
        Returns the chain from oldest to newest.
        """
        chain = []
        current_id = unit_id

        with self._conn() as conn:
            while current_id:
                row = conn.execute(
                    "SELECT * FROM memory_units WHERE id = ?", (current_id,)
                ).fetchone()
                if not row:
                    break
                chain.append(self._row_to_dict(row))
                supersedes = json.loads(row["supersedes_json"])
                current_id = supersedes[0] if supersedes else None

        chain.reverse()
        return chain

    def delete_user_memory(self, user_id: str) -> int:
        """Hard delete all memory for a user (GDPR right to be forgotten)."""
        with self._conn() as conn:
            count = conn.execute(
                "DELETE FROM memory_units WHERE user_id = ?", (user_id,)
            ).rowcount
        return count

    def stats(self, user_id: str) -> dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memory_units WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM memory_units WHERE user_id = ? AND is_superseded = 0",
                (user_id,),
            ).fetchone()[0]
            by_type = conn.execute(
                """
                SELECT type, COUNT(*) as cnt FROM memory_units
                WHERE user_id = ? AND is_superseded = 0
                GROUP BY type
                """,
                (user_id,),
            ).fetchall()
        return {
            "total_units": total,
            "active_units": active,
            "superseded_units": total - active,
            "by_type": {r["type"]: r["cnt"] for r in by_type},
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["tags"] = json.loads(d.pop("tags_json"))
        d["source_turns"] = json.loads(d.pop("source_turns"))
        d["supersedes"] = json.loads(d.pop("supersedes_json"))
        d["is_superseded"] = bool(d["is_superseded"])
        return d
