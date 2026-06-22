//! # multimind
//!
//! **Multi-Model Mind** — a generic ONNX model registry with inference,
//! correction signals, and an optional retrain pipeline for Rust applications.
//!
//! Multimind has **zero knowledge** of any particular product, domain, or
//! storage layer. Consumers wire it into their own routing, storage, and
//! deployment systems.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                 ModelRegistry                    │
//! │  ┌────────────┐  ┌────────────┐                 │
//! │  │ OnnxText   │  │ OnnxEmbed  │  ...custom...   │
//! │  │ (TF-IDF)   │  │ (384-dim)  │                 │
//! │  └─────┬──────┘  └─────┬──────┘                 │
//! │        │ ModelBackend  │                         │
//! │        └───────┬───────┘                         │
//! │                ▼                                 │
//! │         classify(input) → Verdict                │
//! └────────────────┬────────────────────────────────┘
//!                  │ correction signals
//!                  ▼
//! ┌─────────────────────────────────────────────────┐
//! │              SignalStore                         │
//! │  ┌────────────┐  ┌────────────┐                 │
//! │  │  Postgres   │  │   SQLite   │  ...custom...  │
//! │  └─────────────┘  └────────────┘                │
//! └────────────────┬────────────────────────────────┘
//!                  │ batch export
//!                  ▼
//! ┌─────────────────────────────────────────────────┐
//! │            RetrainPipeline (optional)            │
//! │  signals → features → learn → export → hot-swap │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - `sqlite` (default) — SQLite signal store via `rusqlite`
//! - `postgres` — PostgreSQL signal store via `sqlx`
//! - `retrain` — background retrain pipeline with ONNX artifact export
//! - `full` — all of the above
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! multimind = { version = "0.1", features = ["sqlite"] }
//! ```
//!
//! ```rust,no_run
//! use multimind::{ModelRegistry, MultimindConfig, ModelInput};
//!
//! let config = MultimindConfig::from_toml(r#"
//!     [[models]]
//!     id = "classifier"
//!     backend = "onnx-text"
//!     path = "models/classifier.onnx"
//!     labels = "models/labels.json"
//! "#).unwrap();
//!
//! let registry = ModelRegistry::new(config, ".");
//! let verdict = registry.classify("classifier", &ModelInput::Text("hello world".into())).unwrap();
//! println!("{}: {:.2}", verdict.label, verdict.confidence);
//! ```

pub mod backends;
pub mod config;
pub mod registry;
pub mod signals;

#[cfg(feature = "retrain")]
pub mod retrain;

// Re-exports
pub use config::MultimindConfig;
pub use registry::ModelRegistry;

use std::collections::HashMap;
use std::path::Path;

// ── Core traits and types ──────────────────────────────────────────────────

/// Input to a model. Backends accept one or more of these variants.
#[derive(Debug, Clone)]
pub enum ModelInput {
    /// Raw text — for TF-IDF ONNX models that accept string tensors.
    Text(String),
    /// Pre-computed embedding vector — for embedding-based ONNX models.
    Embedding(Vec<f32>),
    /// Structured JSON — for API-backed models (future).
    Structured(serde_json::Value),
}

/// The output of a single model inference call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Verdict {
    /// Winning label (e.g. "store", "episodic", "SAFE").
    pub label: String,
    /// Confidence in the winning label (0.0 – 1.0).
    pub confidence: f32,
    /// Per-class scores (label → probability). Empty if the backend
    /// doesn't support per-class output.
    pub all_scores: HashMap<String, f32>,
}

/// A model backend that can classify inputs.
///
/// Implementations must be `Send + Sync` for use inside `Arc<ModelRegistry>`.
pub trait ModelBackend: Send + Sync {
    /// Run inference on the given input.
    fn classify(&self, input: &ModelInput) -> anyhow::Result<Verdict>;

    /// Hot-reload the model from a new path.
    /// Returns `Ok(())` if reload succeeded, `Err` if not (old model stays loaded).
    fn reload(&self, path: &Path) -> anyhow::Result<()>;

    /// Human-readable backend name (e.g. "onnx-text", "onnx-embed").
    fn backend_name(&self) -> &'static str;
}

// ── Training signals ───────────────────────────────────────────────────────

/// A correction signal for model improvement.
///
/// The consuming service records these; the retrain pipeline reads them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingSignal {
    /// Which model produced the original verdict.
    pub model_id: String,
    /// The input that was classified.
    pub input_text: String,
    /// The model's original prediction.
    pub predicted_label: String,
    /// The corrected label (ground truth from user/system).
    pub corrected_label: String,
    /// Optional confidence of the original prediction.
    pub original_confidence: Option<f32>,
}

/// Storage backend for training signals.
///
/// Sulcus implements this with Postgres. Guardian/GRTE can use SQLite.
/// Consumers can implement custom backends (Redis, JSONL, etc.).
pub trait SignalStore: Send + Sync {
    /// Record a correction signal.
    fn record(&self, signal: &TrainingSignal) -> anyhow::Result<()>;

    /// Count signals for a given model since last retrain.
    fn count_pending(&self, model_id: &str) -> anyhow::Result<usize>;

    /// Export pending signals for retraining.
    fn export_pending(&self, model_id: &str) -> anyhow::Result<Vec<TrainingSignal>>;

    /// Mark signals as consumed (after successful retrain).
    fn mark_consumed(&self, model_id: &str) -> anyhow::Result<()>;
}
