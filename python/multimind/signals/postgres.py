"""PostgreSQL signal store — for production deployments.

Requires ``psycopg2`` (or ``psycopg2-binary``). Install via::

    pip install multimind[postgres]
"""

from __future__ import annotations

import logging

from ..types import TrainingSignal

logger = logging.getLogger(__name__)

MIGRATION_SQL = """\
CREATE TABLE IF NOT EXISTS model_signals (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id            TEXT NOT NULL,
    input_text          TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    corrected_label     TEXT NOT NULL,
    original_confidence REAL,
    consumed            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_signals_pending
    ON model_signals (model_id, consumed, created_at ASC);
"""


class PgSignalStore:
    """PostgreSQL-backed signal store.

    Uses synchronous ``psycopg2`` connections. The consuming service is
    responsible for running the migration (see :data:`MIGRATION_SQL`).

    Args:
        dsn: PostgreSQL connection string (e.g. ``"postgresql://user:pass@host/db"``).
        connection: An existing psycopg2 connection (alternative to dsn).
    """

    def __init__(
        self,
        dsn: str | None = None,
        connection: object | None = None,
    ) -> None:
        if connection is not None:
            self._conn = connection
            self._owns_conn = False
        elif dsn is not None:
            try:
                import psycopg2
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for PgSignalStore. "
                    "Install it with: pip install multimind[postgres]"
                )
            self._conn = psycopg2.connect(dsn)
            self._owns_conn = True
        else:
            raise ValueError("either dsn or connection must be provided")

    def run_migration(self) -> None:
        """Run the migration SQL to create the model_signals table."""
        with self._conn.cursor() as cur:
            cur.execute(MIGRATION_SQL)
        self._conn.commit()

    def record(self, signal: TrainingSignal) -> None:
        """Record a correction signal."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_signals "
                "(model_id, input_text, predicted_label, corrected_label, original_confidence) "
                "VALUES (%s, %s, %s, %s, %s)",
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
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM model_signals WHERE model_id = %s AND consumed = FALSE",
                (model_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def export_pending(
        self, model_id: str, *, limit: int | None = None
    ) -> list[TrainingSignal]:
        """Export pending signals for retraining.

        Args:
            model_id: Model to export signals for.
            limit: Maximum rows to return. ``None`` means all pending.
        """
        with self._conn.cursor() as cur:
            if limit is not None:
                cur.execute(
                    "SELECT id::text, model_id, input_text, predicted_label, corrected_label, original_confidence "
                    "FROM model_signals "
                    "WHERE model_id = %s AND consumed = FALSE "
                    "ORDER BY created_at ASC "
                    "LIMIT %s",
                    (model_id, limit),
                )
            else:
                cur.execute(
                    "SELECT id::text, model_id, input_text, predicted_label, corrected_label, original_confidence "
                    "FROM model_signals "
                    "WHERE model_id = %s AND consumed = FALSE "
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
                    signal_id=row[0],
                )
                for row in cur.fetchall()
            ]

    def mark_consumed(self, model_id: str, signal_ids: list[str]) -> None:
        """Mark specific signals as consumed (after successful retrain)."""
        if not signal_ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE model_signals SET consumed = TRUE "
                "WHERE model_id = %s AND id = ANY(%s::uuid[]) AND consumed = FALSE",
                (model_id, signal_ids),
            )
        self._conn.commit()

    def mark_all_consumed(self, model_id: str) -> None:
        """Mark all pending signals for a model as consumed."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE model_signals SET consumed = TRUE "
                "WHERE model_id = %s AND consumed = FALSE",
                (model_id,),
            )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection (if owned)."""
        if self._owns_conn:
            self._conn.close()
