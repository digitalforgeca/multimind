//! PostgreSQL signal store — for production deployments.
//!
//! Uses `sqlx` with `postgres,runtime-tokio` features.
//! The consuming service is responsible for running the migration
//! (see [`PgSignalStore::MIGRATION_SQL`]).

use sqlx::PgPool;

use crate::{SignalStore, TrainingSignal};

/// PostgreSQL-backed signal store.
///
/// Wraps async sqlx calls via `tokio::task::block_in_place()` +
/// `Handle::block_on()`, which is safe from within an async context
/// on a multi-threaded tokio runtime.
pub struct PgSignalStore {
    pool: PgPool,
}

impl PgSignalStore {
    /// Create a new Postgres signal store.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// SQL migration for the `model_signals` table.
    ///
    /// Run this once at startup or via your migration system.
    pub const MIGRATION_SQL: &'static str = r#"
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
    ON model_signals (model_id, consumed, created_at DESC);
"#;

    /// Run an async future synchronously.
    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
    }
}

impl SignalStore for PgSignalStore {
    fn record(&self, signal: &TrainingSignal) -> anyhow::Result<()> {
        self.block_on(async {
            sqlx::query(
                "INSERT INTO model_signals \
                 (model_id, input_text, predicted_label, corrected_label, original_confidence) \
                 VALUES ($1, $2, $3, $4, $5)",
            )
            .bind(&signal.model_id)
            .bind(&signal.input_text)
            .bind(&signal.predicted_label)
            .bind(&signal.corrected_label)
            .bind(signal.original_confidence)
            .execute(&self.pool)
            .await?;
            Ok(())
        })
    }

    fn count_pending(&self, model_id: &str) -> anyhow::Result<usize> {
        self.block_on(async {
            let row: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM model_signals WHERE model_id = $1 AND consumed = FALSE",
            )
            .bind(model_id)
            .fetch_one(&self.pool)
            .await?;
            Ok(row.0 as usize)
        })
    }

    fn export_pending(
        &self,
        model_id: &str,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<TrainingSignal>> {
        self.block_on(async {
            let rows: Vec<(String, String, String, String, Option<f32>)> = match limit {
                Some(n) => {
                    sqlx::query_as(
                        "SELECT model_id, input_text, predicted_label, corrected_label, original_confidence \
                         FROM model_signals \
                         WHERE model_id = $1 AND consumed = FALSE \
                         ORDER BY created_at ASC \
                         LIMIT $2",
                    )
                    .bind(model_id)
                    .bind(n as i64)
                    .fetch_all(&self.pool)
                    .await?
                }
                None => {
                    sqlx::query_as(
                        "SELECT model_id, input_text, predicted_label, corrected_label, original_confidence \
                         FROM model_signals \
                         WHERE model_id = $1 AND consumed = FALSE \
                         ORDER BY created_at ASC",
                    )
                    .bind(model_id)
                    .fetch_all(&self.pool)
                    .await?
                }
            };

            Ok(rows
                .into_iter()
                .map(|(mid, it, pl, cl, oc)| TrainingSignal {
                    model_id: mid,
                    input_text: it,
                    predicted_label: pl,
                    corrected_label: cl,
                    original_confidence: oc,
                })
                .collect())
        })
    }

    fn mark_consumed(&self, model_id: &str) -> anyhow::Result<()> {
        self.block_on(async {
            sqlx::query(
                "UPDATE model_signals SET consumed = TRUE \
                 WHERE model_id = $1 AND consumed = FALSE",
            )
            .bind(model_id)
            .execute(&self.pool)
            .await?;
            Ok(())
        })
    }
}
