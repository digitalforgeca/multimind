//! Generic retrain pipeline.
//!
//! Consumes correction signals, extracts features, learns weight adjustments,
//! exports ONNX-compatible artifacts, and hot-swaps live models.
//!
//! # Architecture
//!
//! ```text
//! SignalStore (pending signals)
//!   │ export_pending(model_id)
//!   ▼
//! FeatureExtractor → SignalFeatures
//!   │
//!   ▼
//! WeightLearner → WeightModel (updated adjustments)
//!   │
//!   ▼
//! ArtifactExporter → RetrainArtifact (ONNX-compatible binary)
//!   │
//!   ▼
//! ModelRegistry.reload_model() → hot-swap
//! ```
//!
//! # Usage
//!
//! The pipeline is generic over a `WeightModel` type that consumers define.
//! This lets each product (Sulcus, GRTE, etc.) define its own model shape
//! while sharing the pipeline orchestration.
//!
//! ```rust,no_run
//! use multimind::retrain::{RetrainPipeline, RetrainConfig, WeightModel};
//! use multimind::TrainingSignal;
//!
//! // Define your domain-specific weight model
//! #[derive(Clone, serde::Serialize, serde::Deserialize)]
//! struct MyWeights {
//!     adjustments: std::collections::HashMap<String, f64>,
//!     version: u64,
//! }
//!
//! impl WeightModel for MyWeights {
//!     fn version(&self) -> u64 { self.version }
//!     fn set_version(&mut self, v: u64) { self.version = v; }
//!     fn categories(&self) -> Vec<String> { self.adjustments.keys().cloned().collect() }
//!     fn adjustment(&self, category: &str) -> f64 {
//!         self.adjustments.get(category).copied().unwrap_or(1.0)
//!     }
//!     fn set_adjustment(&mut self, category: &str, value: f64) {
//!         self.adjustments.insert(category.to_string(), value);
//!     }
//! }
//! ```

pub mod pipeline;
pub mod types;

pub use pipeline::RetrainPipeline;
pub use types::*;
