"""SQLite signal store — for lightweight deployments, CLI tools, and testing."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ..types import TrainingSignal

_MIGRATION_SQL = """\
CREATE TABLE IF NOT EXISTS model_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id            TEXT NOT NULL,
    input_text          TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    corrected_label     TEXT NOT NULL,
    original_confidence REAL,
    consumed            INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_model_signals_pending
    ON model_signals (model_id, consumed, created_at);
"""


class SqliteSignalStore:
    """SQLite-backed signal store.

    Thread-safe via internal lock. Suitable for single-process tools,
    desktop apps, CLI analyzers, and tests.

    Args:
        path: Path to the SQLite database file, or ``":memory:"`` for in-memory.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_MIGRATION_SQL)

    @classmethod
    def open(cls, path: str | Path) -> SqliteSignalStore:
        """Open (or create) a SQLite signal store at the given path."""
        return cls(path)

    @classmethod
    def in_memory(cls) -> SqliteSignalStore:
        """Create an in-memory signal store (for tests)."""
        return cls(":memory:")

    def record(self, signal: TrainingSignal) -> None:
        """Record a correction signal."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO model_signals "
                "(model_id, input_text, predicted_label, corrected_label, original_confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    signal.model_id,
                    signal.input_text,
                    signal.predicted_label,
                    signal.corrected_label,
                    signal.original_confidence,
                ),
            )
            self._conn.commit()

    def count_pending(self, model_id: str) -> int:
        """Count signals for a given model since last retrain."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM model_signals WHERE model_id = ? AND consumed = 0",
                (model_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def export_pending(
        self, model_id: str, *, limit: int | None = None
    ) -> list[TrainingSignal]:
        """Export pending signals for retraining.

        Args:
            model_id: Model to export signals for.
            limit: Maximum rows to return. ``None`` means all pending.
        """
        with self._lock:
            if limit is not None:
                cursor = self._conn.execute(
                    "SELECT id, model_id, input_text, predicted_label, corrected_label, original_confidence "
                    "FROM model_signals "
                    "WHERE model_id = ? AND consumed = 0 "
                    "ORDER BY created_at ASC "
                    "LIMIT ?",
                    (model_id, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, model_id, input_text, predicted_label, corrected_label, original_confidence "
                    "FROM model_signals "
                    "WHERE model_id = ? AND consumed = 0 "
                    "ORDER BY created_at ASC",
                    (model_id,),
                )
            return [
                TrainingSignal(
                    model_id=row[1],
                    input_text=row[2],
                    predicted_label=row[3],
                    corrected_label=row[4],
                    original_confidence=row[5],
                    signal_id=str(row[0]),
                )
                for row in cursor.fetchall()
            ]

    def mark_consumed(self, model_id: str, signal_ids: list[str]) -> None:
        """Mark specific signals as consumed (after successful retrain)."""
        if not signal_ids:
            return
        with self._lock:
            placeholders = ", ".join("?" for _ in signal_ids)
            self._conn.execute(
                f"UPDATE model_signals SET consumed = 1 "
                f"WHERE model_id = ? AND id IN ({placeholders}) AND consumed = 0",
                [model_id] + [int(sid) for sid in signal_ids],
            )
            self._conn.commit()

    def mark_all_consumed(self, model_id: str) -> None:
        """Mark all pending signals for a model as consumed."""
        with self._lock:
            self._conn.execute(
                "UPDATE model_signals SET consumed = 1 WHERE model_id = ? AND consumed = 0",
                (model_id,),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
