//! Retrain pipeline types — traits and structs for the generic pipeline.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::TrainingSignal;

// ── Configuration ──────────────────────────────────────────────────────────

/// Retrain pipeline configuration.
#[derive(Debug, Clone)]
pub struct RetrainConfig {
    /// Minimum unconsumed signals before a retrain is eligible.
    pub signal_threshold: usize,
    /// Maximum signals to consume per retrain batch.
    pub batch_size: usize,
    /// Background check interval (how often to poll for threshold).
    pub check_interval: Duration,
    /// Learning rate for weight updates (0.0–1.0).
    pub learning_rate: f64,
    /// Minimum correction signals for a category before applying its update.
    /// Prevents single-signal noise from shifting weights.
    pub min_corrections_for_update: usize,
    /// Directory for persisting model artifacts.
    pub artifact_dir: String,
}

impl Default for RetrainConfig {
    fn default() -> Self {
        Self {
            signal_threshold: 200,
            batch_size: 1000,
            check_interval: Duration::from_secs(3600),
            learning_rate: 0.05,
            min_corrections_for_update: 5,
            artifact_dir: "/tmp/multimind-models".into(),
        }
    }
}

// ── Weight model trait ─────────────────────────────────────────────────────

/// A learnable weight model.
///
/// Consumers implement this for their domain-specific model shape.
/// The retrain pipeline operates on this trait generically.
///
/// Note: [`RetrainArtifact::from_model`] only requires the trait methods,
/// not `Serialize`/`Deserialize`. If your model needs serialization for
/// persistence, implement serde on your concrete type.
pub trait WeightModel: Clone + Send + Sync {
    /// Model version (monotonically increasing).
    fn version(&self) -> u64;

    /// Set the model version.
    fn set_version(&mut self, version: u64);

    /// All category names this model tracks.
    fn categories(&self) -> Vec<String>;

    /// Get the weight adjustment for a category (1.0 = no change).
    fn adjustment(&self, category: &str) -> f64;

    /// Set the weight adjustment for a category.
    fn set_adjustment(&mut self, category: &str, value: f64);
}

// ── Feature extraction ─────────────────────────────────────────────────────

/// Extracted features from a batch of training signals.
#[derive(Debug, Clone)]
pub struct SignalFeatures {
    /// Total signals in the batch.
    pub total: usize,
    /// Per-category signal counts and correction rates.
    pub category_signals: HashMap<String, CategoryFeatures>,
}

/// Features for a single category.
#[derive(Debug, Clone)]
pub struct CategoryFeatures {
    /// Total signals where this category was involved.
    pub total: usize,
    /// Signals where the model prediction was correct.
    pub correct: usize,
    /// Signals where the model prediction was wrong (corrections).
    pub corrections: usize,
    /// Average confidence on correct predictions.
    pub avg_confidence_correct: f64,
    /// Average confidence on incorrect predictions.
    pub avg_confidence_incorrect: f64,
}

/// Extract features from a batch of training signals.
///
/// This is domain-agnostic: it groups by predicted/corrected labels
/// and computes correction rates and confidence distributions.
pub fn extract_features(signals: &[TrainingSignal]) -> SignalFeatures {
    let mut category_map: HashMap<String, (usize, usize, usize, f64, f64)> = HashMap::new();

    for signal in signals {
        let is_correct = signal.predicted_label == signal.corrected_label;
        let confidence = signal.original_confidence.unwrap_or(0.5) as f64;

        // Count for predicted category
        let entry = category_map
            .entry(signal.predicted_label.clone())
            .or_insert((0, 0, 0, 0.0, 0.0));
        entry.0 += 1; // total
        if is_correct {
            entry.1 += 1; // correct
            entry.3 += confidence; // sum correct confidence
        } else {
            entry.2 += 1; // corrections
            entry.4 += confidence; // sum incorrect confidence
        }

        // Also count for corrected category (if different)
        if !is_correct {
            let corrected_entry = category_map
                .entry(signal.corrected_label.clone())
                .or_insert((0, 0, 0, 0.0, 0.0));
            corrected_entry.0 += 1;
            corrected_entry.1 += 1; // The correction itself is "correct" for the target
            corrected_entry.3 += confidence;
        }
    }

    let category_signals = category_map
        .into_iter()
        .map(|(cat, (total, correct, corrections, sum_conf_correct, sum_conf_incorrect))| {
            (
                cat,
                CategoryFeatures {
                    total,
                    correct,
                    corrections,
                    avg_confidence_correct: if correct > 0 {
                        sum_conf_correct / correct as f64
                    } else {
                        0.0
                    },
                    avg_confidence_incorrect: if corrections > 0 {
                        sum_conf_incorrect / corrections as f64
                    } else {
                        0.0
                    },
                },
            )
        })
        .collect();

    SignalFeatures {
        total: signals.len(),
        category_signals,
    }
}

/// Apply learned weight updates to a model based on extracted features.
///
/// Uses the correction rate to adjust category weights:
/// - High correction rate → suppress (lower weight)
/// - Low correction rate → boost (higher weight)
///
/// Only applies updates when a category has enough corrections
/// (controlled by `min_corrections_for_update`).
pub fn learn_weights<M: WeightModel>(
    model: &M,
    features: &SignalFeatures,
    config: &RetrainConfig,
) -> M {
    let mut updated = model.clone();
    updated.set_version(model.version() + 1);

    for category in model.categories() {
        if let Some(cat_features) = features.category_signals.get(&category) {
            // Skip if not enough corrections to be statistically meaningful
            if cat_features.corrections < config.min_corrections_for_update {
                continue;
            }

            let correction_rate = if cat_features.total > 0 {
                cat_features.corrections as f64 / cat_features.total as f64
            } else {
                0.0
            };

            // Adjustment: high correction rate → reduce weight, low → increase
            // correction_rate of 0.5 means 50% of predictions are wrong → significant decrease
            let current = model.adjustment(&category);
            let delta = config.learning_rate * (1.0 - 2.0 * correction_rate);
            let new_adjustment = (current + delta).clamp(0.1, 10.0);

            updated.set_adjustment(&category, new_adjustment);
        }
    }

    updated
}

// ── Artifact ───────────────────────────────────────────────────────────────

/// A retrain artifact — the output of a successful retrain cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainArtifact {
    /// Model ID this artifact is for.
    pub model_id: String,
    /// Model version.
    pub version: u64,
    /// Category names (row/column order for the weight matrix).
    pub categories: Vec<String>,
    /// Flattened NxN diagonal weight matrix (row-major).
    /// Encodes per-category adjustment factors on the diagonal.
    pub weight_matrix: Vec<f32>,
    /// SHA-256 checksum of the weight matrix bytes.
    pub checksum: String,
    /// When this artifact was created.
    pub created_at: DateTime<Utc>,
    /// Number of signals consumed to produce this artifact.
    pub signals_consumed: usize,
}

impl RetrainArtifact {
    /// Create an artifact from a weight model.
    pub fn from_model<M: WeightModel>(model: &M, model_id: &str, signals_consumed: usize) -> Self {
        let categories = model.categories();
        let n = categories.len();

        // Build diagonal weight matrix
        let mut weight_matrix = vec![0.0f32; n * n];
        for (i, cat) in categories.iter().enumerate() {
            weight_matrix[i * n + i] = model.adjustment(cat) as f32;
        }

        // Compute checksum
        let matrix_bytes: Vec<u8> = weight_matrix.iter().flat_map(|f| f.to_le_bytes()).collect();
        use sha2::{Digest, Sha256};
        let checksum = format!("{:x}", Sha256::digest(&matrix_bytes));

        Self {
            model_id: model_id.to_string(),
            version: model.version(),
            categories,
            weight_matrix,
            checksum,
            created_at: Utc::now(),
            signals_consumed,
        }
    }

    /// Verify the integrity of this artifact.
    pub fn verify(&self) -> bool {
        let matrix_bytes: Vec<u8> = self
            .weight_matrix
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        use sha2::{Digest, Sha256};
        let computed = format!("{:x}", Sha256::digest(&matrix_bytes));
        computed == self.checksum
    }

    /// Persist the artifact to disk as JSON.
    pub fn save(&self, dir: &str) -> anyhow::Result<std::path::PathBuf> {
        let dir_path = std::path::Path::new(dir);
        std::fs::create_dir_all(dir_path)?;

        let filename = format!("{}_v{}.json", self.model_id, self.version);
        let path = dir_path.join(&filename);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, &json)?;

        // Also write a "latest" pointer
        let latest_path = dir_path.join(format!("{}_latest.json", self.model_id));
        std::fs::write(&latest_path, &json)?;

        tracing::info!(
            model_id = %self.model_id,
            version = self.version,
            path = %path.display(),
            "RetrainArtifact: saved"
        );

        Ok(path)
    }

    /// Load an artifact from a JSON file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let artifact: Self = serde_json::from_str(&json)?;
        if !artifact.verify() {
            anyhow::bail!(
                "artifact integrity check failed for {} v{}",
                artifact.model_id,
                artifact.version
            );
        }
        Ok(artifact)
    }
}

// ── Retrain result / status ────────────────────────────────────────────────

/// Result of a retrain cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainResult {
    /// Model ID that was retrained.
    pub model_id: String,
    /// New model version.
    pub new_version: u64,
    /// Previous model version.
    pub previous_version: u64,
    /// Number of signals consumed.
    pub signals_consumed: usize,
    /// Path to the saved artifact (if persisted).
    pub artifact_path: Option<String>,
    /// Duration of the retrain cycle.
    pub duration_ms: u64,
}

/// Current status of the retrain pipeline for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainStatus {
    /// Current model version.
    pub model_version: u64,
    /// Number of unconsumed signals.
    pub unconsumed_signals: usize,
    /// Whether the signal threshold is met for retraining.
    pub threshold_met: bool,
    /// Whether a retrain is currently running.
    pub running: bool,
    /// Last retrain result (if any).
    pub last_result: Option<RetrainResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_features_empty() {
        let features = extract_features(&[]);
        assert_eq!(features.total, 0);
        assert!(features.category_signals.is_empty());
    }

    #[test]
    fn extract_features_correct_predictions() {
        let signals = vec![
            TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: "hello".into(),
                predicted_label: "safe".into(),
                corrected_label: "safe".into(),
                original_confidence: Some(0.9),
            },
            TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: "world".into(),
                predicted_label: "safe".into(),
                corrected_label: "safe".into(),
                original_confidence: Some(0.8),
            },
        ];

        let features = extract_features(&signals);
        assert_eq!(features.total, 2);
        let safe = &features.category_signals["safe"];
        assert_eq!(safe.total, 2);
        assert_eq!(safe.correct, 2);
        assert_eq!(safe.corrections, 0);
    }

    #[test]
    fn extract_features_with_corrections() {
        let signals = vec![
            TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: "pii data".into(),
                predicted_label: "safe".into(),
                corrected_label: "unsafe".into(),
                original_confidence: Some(0.7),
            },
        ];

        let features = extract_features(&signals);
        assert_eq!(features.total, 1);
        let safe = &features.category_signals["safe"];
        assert_eq!(safe.corrections, 1);
        assert!(features.category_signals.contains_key("unsafe"));
    }

    #[test]
    fn learn_weights_no_change_below_threshold() {
        #[derive(Clone, Serialize, Deserialize)]
        struct TestModel {
            version: u64,
            adjustments: HashMap<String, f64>,
        }

        impl WeightModel for TestModel {
            fn version(&self) -> u64 {
                self.version
            }
            fn set_version(&mut self, v: u64) {
                self.version = v;
            }
            fn categories(&self) -> Vec<String> {
                self.adjustments.keys().cloned().collect()
            }
            fn adjustment(&self, cat: &str) -> f64 {
                self.adjustments.get(cat).copied().unwrap_or(1.0)
            }
            fn set_adjustment(&mut self, cat: &str, val: f64) {
                self.adjustments.insert(cat.to_string(), val);
            }
        }

        let model = TestModel {
            version: 0,
            adjustments: [("safe".into(), 1.0), ("unsafe".into(), 1.0)]
                .into_iter()
                .collect(),
        };

        // Only 2 corrections — below default min_corrections_for_update of 5
        let signals = vec![
            TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: "a".into(),
                predicted_label: "safe".into(),
                corrected_label: "unsafe".into(),
                original_confidence: Some(0.6),
            },
            TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: "b".into(),
                predicted_label: "safe".into(),
                corrected_label: "unsafe".into(),
                original_confidence: Some(0.5),
            },
        ];

        let features = extract_features(&signals);
        let config = RetrainConfig::default();
        let updated = learn_weights(&model, &features, &config);

        // Version should bump
        assert_eq!(updated.version, 1);
        // But adjustments shouldn't change (below threshold)
        assert!((updated.adjustment("safe") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn learn_weights_applies_above_threshold() {
        #[derive(Clone, Serialize, Deserialize)]
        struct TestModel {
            version: u64,
            adjustments: HashMap<String, f64>,
        }

        impl WeightModel for TestModel {
            fn version(&self) -> u64 {
                self.version
            }
            fn set_version(&mut self, v: u64) {
                self.version = v;
            }
            fn categories(&self) -> Vec<String> {
                self.adjustments.keys().cloned().collect()
            }
            fn adjustment(&self, cat: &str) -> f64 {
                self.adjustments.get(cat).copied().unwrap_or(1.0)
            }
            fn set_adjustment(&mut self, cat: &str, val: f64) {
                self.adjustments.insert(cat.to_string(), val);
            }
        }

        let model = TestModel {
            version: 0,
            adjustments: [("safe".into(), 1.0), ("unsafe".into(), 1.0)]
                .into_iter()
                .collect(),
        };

        // 10 corrections (safe → unsafe) — well above min_corrections_for_update of 5
        let signals: Vec<_> = (0..10)
            .map(|i| TrainingSignal {
                signal_id: None,
                model_id: "test".into(),
                input_text: format!("input {i}"),
                predicted_label: "safe".into(),
                corrected_label: "unsafe".into(),
                original_confidence: Some(0.6),
            })
            .collect();

        let features = extract_features(&signals);
        let config = RetrainConfig::default();
        let updated = learn_weights(&model, &features, &config);

        // Version should bump
        assert_eq!(updated.version, 1);
        // "safe" had 100% correction rate → should decrease
        assert!(updated.adjustment("safe") < 1.0);
    }

    #[test]
    fn artifact_save_load_round_trip() {
        #[derive(Clone, Serialize, Deserialize)]
        struct TestModel {
            version: u64,
            adjustments: HashMap<String, f64>,
        }

        impl WeightModel for TestModel {
            fn version(&self) -> u64 {
                self.version
            }
            fn set_version(&mut self, v: u64) {
                self.version = v;
            }
            fn categories(&self) -> Vec<String> {
                self.adjustments.keys().cloned().collect()
            }
            fn adjustment(&self, cat: &str) -> f64 {
                self.adjustments.get(cat).copied().unwrap_or(1.0)
            }
            fn set_adjustment(&mut self, cat: &str, val: f64) {
                self.adjustments.insert(cat.to_string(), val);
            }
        }

        let model = TestModel {
            version: 3,
            adjustments: [("x".into(), 0.9), ("y".into(), 1.3)]
                .into_iter()
                .collect(),
        };

        let artifact = RetrainArtifact::from_model(&model, "roundtrip_test", 42);
        let dir = std::env::temp_dir()
            .join(format!("multimind_test_{}", std::process::id()));
        let saved_path = artifact.save(dir.to_str().unwrap()).unwrap();

        // Load it back
        let loaded = RetrainArtifact::load(&saved_path).unwrap();
        assert_eq!(loaded.model_id, "roundtrip_test");
        assert_eq!(loaded.version, 3);
        assert_eq!(loaded.signals_consumed, 42);
        assert!(loaded.verify());
        assert_eq!(loaded.checksum, artifact.checksum);

        // Also verify the "latest" pointer was written
        let latest_path = dir.join("roundtrip_test_latest.json");
        let latest = RetrainArtifact::load(&latest_path).unwrap();
        assert_eq!(latest.version, 3);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn artifact_integrity() {
        #[derive(Clone, Serialize, Deserialize)]
        struct TestModel {
            version: u64,
            adjustments: HashMap<String, f64>,
        }

        impl WeightModel for TestModel {
            fn version(&self) -> u64 {
                self.version
            }
            fn set_version(&mut self, v: u64) {
                self.version = v;
            }
            fn categories(&self) -> Vec<String> {
                self.adjustments.keys().cloned().collect()
            }
            fn adjustment(&self, cat: &str) -> f64 {
                self.adjustments.get(cat).copied().unwrap_or(1.0)
            }
            fn set_adjustment(&mut self, cat: &str, val: f64) {
                self.adjustments.insert(cat.to_string(), val);
            }
        }

        let model = TestModel {
            version: 1,
            adjustments: [("a".into(), 0.8), ("b".into(), 1.2)]
                .into_iter()
                .collect(),
        };

        let artifact = RetrainArtifact::from_model(&model, "test", 100);
        assert!(artifact.verify());
        assert_eq!(artifact.version, 1);
        assert_eq!(artifact.signals_consumed, 100);
    }
}

/// Integration tests that exercise the full pipeline with a real signal store.
#[cfg(all(test, feature = "sqlite"))]
mod integration_tests {
    use super::*;
    use crate::SignalStore;
    use crate::signals::sqlite::SqliteSignalStore;
    use crate::retrain::pipeline::RetrainPipeline;

    #[derive(Clone, Serialize, Deserialize)]
    struct TestWeights {
        version: u64,
        adjustments: HashMap<String, f64>,
    }

    impl WeightModel for TestWeights {
        fn version(&self) -> u64 { self.version }
        fn set_version(&mut self, v: u64) { self.version = v; }
        fn categories(&self) -> Vec<String> { self.adjustments.keys().cloned().collect() }
        fn adjustment(&self, cat: &str) -> f64 { self.adjustments.get(cat).copied().unwrap_or(1.0) }
        fn set_adjustment(&mut self, cat: &str, val: f64) { self.adjustments.insert(cat.to_string(), val); }
    }

    #[test]
    fn pipeline_run_retrain_end_to_end() {
        let store = SqliteSignalStore::in_memory().unwrap();

        // Seed 10 correction signals
        for i in 0..10 {
            store.record(&TrainingSignal {
                signal_id: None,
                model_id: "test_pipeline".into(),
                input_text: format!("input {i}"),
                predicted_label: "safe".into(),
                corrected_label: "unsafe".into(),
                original_confidence: Some(0.65),
            }).unwrap();
        }

        assert_eq!(store.count_pending("test_pipeline").unwrap(), 10);

        let config = RetrainConfig {
            signal_threshold: 5,
            batch_size: 100,
            min_corrections_for_update: 3,
            artifact_dir: std::env::temp_dir()
                .join(format!("multimind_pipeline_test_{}", std::process::id()))
                .to_string_lossy()
                .to_string(),
            ..Default::default()
        };

        let baseline = TestWeights {
            version: 0,
            adjustments: [("safe".into(), 1.0), ("unsafe".into(), 1.0)]
                .into_iter()
                .collect(),
        };

        let pipeline = RetrainPipeline::new(config.clone(), "test_pipeline", baseline);

        // Verify initial state
        assert_eq!(pipeline.current_model().version(), 0);
        assert!(pipeline.latest_artifact().is_none());

        // Run retrain
        let result = pipeline.run_retrain(&store, None).unwrap();

        assert_eq!(result.model_id, "test_pipeline");
        assert_eq!(result.new_version, 1);
        assert_eq!(result.previous_version, 0);
        assert_eq!(result.signals_consumed, 10);
        assert!(result.artifact_path.is_some());

        // Model should be updated
        assert_eq!(pipeline.current_model().version(), 1);
        assert!(pipeline.latest_artifact().is_some());

        // Signals should be consumed
        assert_eq!(store.count_pending("test_pipeline").unwrap(), 0);

        // Status should reflect new state
        let status = pipeline.status(0);
        assert_eq!(status.model_version, 1);
        assert!(!status.running);
        assert!(status.last_result.is_some());

        // Cleanup
        let _ = std::fs::remove_dir_all(&config.artifact_dir);
    }

    #[test]
    fn pipeline_run_retrain_no_signals() {
        let store = SqliteSignalStore::in_memory().unwrap();
        let config = RetrainConfig::default();
        let baseline = TestWeights {
            version: 0,
            adjustments: HashMap::new(),
        };

        let pipeline = RetrainPipeline::new(config, "empty_model", baseline);
        let result = pipeline.run_retrain(&store, None);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no pending signals");
        // Running flag should be reset even on error
        assert!(!pipeline.is_running());
    }
}
