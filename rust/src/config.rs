//! TOML-based configuration for the model registry.
//!
//! # Example
//!
//! ```toml
//! [[models]]
//! id = "classifier"
//! backend = "onnx-text"
//! path = "models/classifier.onnx"
//! labels = "models/labels.json"
//! classes = ["safe", "unsafe"]
//!
//! [[models]]
//! id = "embedder"
//! backend = "onnx-embed"
//! path = "models/embed_classifier.onnx"
//! labels = "models/embed_labels.json"
//! embedding_dim = 384
//!
//! [models.retrain]
//! min_signals = 50
//! min_sessions = 100
//! ```

use serde::{Deserialize, Serialize};

/// Top-level multimind configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimindConfig {
    /// Registered models.
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

/// Configuration for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Unique identifier for this model (e.g. "sivu", "vibeguard").
    pub id: String,

    /// Backend type: "onnx-text", "onnx-embed", or a custom backend name.
    pub backend: String,

    /// Path to the model file (ONNX, GGUF, etc.). Relative to model root.
    pub path: String,

    /// Path to the label map JSON file. Optional — some backends have defaults.
    pub labels: Option<String>,

    /// Expected class names. Optional — derived from labels file if not set.
    pub classes: Option<Vec<String>>,

    /// Minimum confidence to accept a classification. Default: 0.5.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,

    /// Number of embedding dimensions. Only meaningful for `onnx-embed` backend.
    /// Default: 384 when not specified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dim: Option<usize>,

    /// Retrain configuration. Optional.
    pub retrain: Option<ModelRetrainConfig>,
}

/// Per-model retrain thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetrainConfig {
    /// Minimum number of correction signals before retraining.
    #[serde(default = "default_min_signals")]
    pub min_signals: usize,

    /// Minimum number of classification sessions before retraining.
    #[serde(default = "default_min_sessions")]
    pub min_sessions: usize,
}

fn default_min_confidence() -> f32 {
    0.5
}
fn default_min_signals() -> usize {
    10
}
fn default_min_sessions() -> usize {
    20
}

impl MultimindConfig {
    /// Parse from a TOML string.
    pub fn from_toml(toml_str: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(toml_str)?)
    }

    /// Parse from a TOML file path.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_toml(&contents)
    }

    /// Find a model config by ID.
    pub fn get_model(&self, id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let toml = r#"
            [[models]]
            id = "classifier"
            backend = "onnx-text"
            path = "models/classifier.onnx"
        "#;
        let config = MultimindConfig::from_toml(toml).unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "classifier");
        assert_eq!(config.models[0].backend, "onnx-text");
        assert_eq!(config.models[0].min_confidence, 0.5);
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
            [[models]]
            id = "vibeguard"
            backend = "onnx-text"
            path = "models/vibeguard.onnx"
            labels = "models/vibeguard_labels.json"
            classes = ["SAFE", "UNSAFE", "REVIEW"]
            min_confidence = 0.7

            [models.retrain]
            min_signals = 50
            min_sessions = 100
        "#;
        let config = MultimindConfig::from_toml(toml).unwrap();
        let m = &config.models[0];
        assert_eq!(m.id, "vibeguard");
        assert_eq!(m.classes.as_ref().unwrap().len(), 3);
        assert_eq!(m.min_confidence, 0.7);
        assert_eq!(m.retrain.as_ref().unwrap().min_signals, 50);
    }

    #[test]
    fn parse_multi_model_config() {
        let toml = r#"
            [[models]]
            id = "sivu"
            backend = "onnx-text"
            path = "models/sivu.onnx"

            [[models]]
            id = "sicu"
            backend = "onnx-embed"
            path = "models/sicu.onnx"
            embedding_dim = 384
        "#;
        let config = MultimindConfig::from_toml(toml).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.get_model("sivu").unwrap().backend, "onnx-text");
        assert_eq!(config.get_model("sicu").unwrap().embedding_dim, Some(384));
        assert!(config.get_model("nonexistent").is_none());
    }
}
