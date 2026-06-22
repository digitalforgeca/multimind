//! SQLite signal store — for lightweight deployments, CLI tools, and testing.

use std::path::PathBuf;
use std::sync::Mutex;

use rusqlite::Connection;

use crate::{SignalStore, TrainingSignal};

/// SQLite-backed signal store.
///
/// Thread-safe via internal Mutex. Suitable for single-process tools,
/// desktop apps, CLI analyzers, and tests.
pub struct SqliteSignalStore {
    conn: Mutex<Connection>,
}

impl SqliteSignalStore {
    /// Open (or create) a SQLite signal store at the given path.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, rusqlite::Error> {
        let path = path.into();
        let conn = Connection::open(&path)?;
        conn.execute_batch(Self::MIGRATION_SQL)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Create an in-memory signal store (for tests).
    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(Self::MIGRATION_SQL)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    const MIGRATION_SQL: &'static str = r#"
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
"#;
}

impl SignalStore for SqliteSignalStore {
    fn record(&self, signal: &TrainingSignal) -> anyhow::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        conn.execute(
            "INSERT INTO model_signals \
             (model_id, input_text, predicted_label, corrected_label, original_confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                signal.model_id,
                signal.input_text,
                signal.predicted_label,
                signal.corrected_label,
                signal.original_confidence,
            ],
        )?;
        Ok(())
    }

    fn count_pending(&self, model_id: &str) -> anyhow::Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM model_signals WHERE model_id = ?1 AND consumed = 0",
            rusqlite::params![model_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    fn export_pending(
        &self,
        model_id: &str,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<TrainingSignal>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;

        let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match limit {
            Some(n) => (
                "SELECT model_id, input_text, predicted_label, corrected_label, original_confidence \
                 FROM model_signals \
                 WHERE model_id = ?1 AND consumed = 0 \
                 ORDER BY created_at ASC \
                 LIMIT ?2"
                    .to_string(),
                vec![Box::new(model_id.to_string()), Box::new(n as i64)],
            ),
            None => (
                "SELECT model_id, input_text, predicted_label, corrected_label, original_confidence \
                 FROM model_signals \
                 WHERE model_id = ?1 AND consumed = 0 \
                 ORDER BY created_at ASC"
                    .to_string(),
                vec![Box::new(model_id.to_string())],
            ),
        };

        let mut stmt = conn.prepare(&sql)?;
        let signals = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                Ok(TrainingSignal {
                    model_id: row.get(0)?,
                    input_text: row.get(1)?,
                    predicted_label: row.get(2)?,
                    corrected_label: row.get(3)?,
                    original_confidence: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(signals)
    }

    fn mark_consumed(&self, model_id: &str) -> anyhow::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        conn.execute(
            "UPDATE model_signals SET consumed = 1 WHERE model_id = ?1 AND consumed = 0",
            rusqlite::params![model_id],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let store = SqliteSignalStore::in_memory().unwrap();

        let signal = TrainingSignal {
            model_id: "test_model".to_string(),
            input_text: "test content".to_string(),
            predicted_label: "reject".to_string(),
            corrected_label: "store".to_string(),
            original_confidence: Some(0.52),
        };

        store.record(&signal).unwrap();
        assert_eq!(store.count_pending("test_model").unwrap(), 1);
        assert_eq!(store.count_pending("other_model").unwrap(), 0);

        let signals = store.export_pending("test_model", None).unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].predicted_label, "reject");
        assert_eq!(signals[0].corrected_label, "store");

        store.mark_consumed("test_model").unwrap();
        assert_eq!(store.count_pending("test_model").unwrap(), 0);
    }

    #[test]
    fn multiple_models_isolated() {
        let store = SqliteSignalStore::in_memory().unwrap();

        for i in 0..3 {
            store
                .record(&TrainingSignal {
                    model_id: "model_a".to_string(),
                    input_text: format!("input {i}"),
                    predicted_label: "class_1".to_string(),
                    corrected_label: "class_2".to_string(),
                    original_confidence: Some(0.6),
                })
                .unwrap();
        }

        store
            .record(&TrainingSignal {
                model_id: "model_b".to_string(),
                input_text: "another input".to_string(),
                predicted_label: "x".to_string(),
                corrected_label: "y".to_string(),
                original_confidence: None,
            })
            .unwrap();

        assert_eq!(store.count_pending("model_a").unwrap(), 3);
        assert_eq!(store.count_pending("model_b").unwrap(), 1);

        // Consuming one model doesn't affect the other
        store.mark_consumed("model_a").unwrap();
        assert_eq!(store.count_pending("model_a").unwrap(), 0);
        assert_eq!(store.count_pending("model_b").unwrap(), 1);
    }

    #[test]
    fn export_with_limit() {
        let store = SqliteSignalStore::in_memory().unwrap();

        for i in 0..10 {
            store
                .record(&TrainingSignal {
                    model_id: "test".to_string(),
                    input_text: format!("input {i}"),
                    predicted_label: "a".to_string(),
                    corrected_label: "b".to_string(),
                    original_confidence: Some(0.5),
                })
                .unwrap();
        }

        // Without limit: get all 10
        let all = store.export_pending("test", None).unwrap();
        assert_eq!(all.len(), 10);

        // With limit: get exactly 3
        let limited = store.export_pending("test", Some(3)).unwrap();
        assert_eq!(limited.len(), 3);

        // Limit larger than available: get all
        let big_limit = store.export_pending("test", Some(100)).unwrap();
        assert_eq!(big_limit.len(), 10);
    }
}
