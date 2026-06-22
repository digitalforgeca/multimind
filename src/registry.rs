//! Model registry — load, manage, and query models by ID.
//!
//! Models are loaded lazily on first `classify()` call and cached.
//! Hot-reload replaces the model in-place without dropping the registry.
//! Custom backends can be registered alongside the built-in ONNX backends.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Context};

use crate::backends::onnx_embed::OnnxEmbedBackend;
use crate::backends::onnx_text::OnnxTextBackend;
use crate::config::{ModelConfig, MultimindConfig};
use crate::{ModelBackend, ModelInput, Verdict};

/// A loaded model instance.
struct LoadedModel {
    backend: Box<dyn ModelBackend>,
}

/// Factory function for creating custom backends.
///
/// Receives the model config and the resolved model root path.
/// Return `None` to fall through to built-in backend resolution.
pub type BackendFactory =
    Box<dyn Fn(&ModelConfig, &Path) -> anyhow::Result<Option<Box<dyn ModelBackend>>> + Send + Sync>;

/// The model registry. Thread-safe, supports lazy loading and hot-reload.
pub struct ModelRegistry {
    config: RwLock<MultimindConfig>,
    model_root: PathBuf,
    models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    custom_factories: Vec<BackendFactory>,
}

impl ModelRegistry {
    /// Create a new registry from config. Models are NOT loaded yet — they
    /// load lazily on first `classify()` call.
    pub fn new(config: MultimindConfig, model_root: impl Into<PathBuf>) -> Self {
        Self {
            config: RwLock::new(config),
            model_root: model_root.into(),
            models: RwLock::new(HashMap::new()),
            custom_factories: Vec::new(),
        }
    }

    /// Create from a TOML config file.
    pub fn from_file(config_path: &Path, model_root: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let config = MultimindConfig::from_file(config_path)?;
        Ok(Self::new(config, model_root))
    }

    /// Register a custom backend factory.
    ///
    /// Factories are tried in registration order before the built-in backends.
    /// If a factory returns `Ok(Some(backend))`, that backend is used.
    /// If it returns `Ok(None)`, the next factory (or built-in) is tried.
    pub fn register_backend_factory(&mut self, factory: BackendFactory) {
        self.custom_factories.push(factory);
    }

    /// List all registered model IDs (from config, whether loaded or not).
    pub fn model_ids(&self) -> Vec<String> {
        self.config
            .read()
            .map(|c| c.models.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default()
    }

    /// Check if a model is currently loaded (vs. just configured).
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.models
            .read()
            .map(|m| m.contains_key(model_id))
            .unwrap_or(false)
    }

    /// Classify input using a specific model. Loads the model lazily if needed.
    pub fn classify(&self, model_id: &str, input: &ModelInput) -> anyhow::Result<Verdict> {
        // Fast path: model already loaded
        {
            let models = self
                .models
                .read()
                .map_err(|_| anyhow!("models RwLock poisoned"))?;
            if let Some(loaded) = models.get(model_id) {
                return loaded.backend.classify(input);
            }
        }

        // Slow path: load the model
        self.load_model(model_id)?;

        let models = self
            .models
            .read()
            .map_err(|_| anyhow!("models RwLock poisoned"))?;
        models
            .get(model_id)
            .ok_or_else(|| anyhow!("model '{}' failed to load", model_id))?
            .backend
            .classify(input)
    }

    /// Explicitly load a model by ID. No-op if already loaded.
    pub fn load_model(&self, model_id: &str) -> anyhow::Result<()> {
        let mut models = self
            .models
            .write()
            .map_err(|_| anyhow!("models RwLock poisoned"))?;

        if models.contains_key(model_id) {
            return Ok(());
        }

        let config = self
            .config
            .read()
            .map_err(|_| anyhow!("config RwLock poisoned"))?;

        let model_config = config
            .get_model(model_id)
            .ok_or_else(|| anyhow!("no model with id '{}' in config", model_id))?
            .clone();

        drop(config); // Release read lock before loading

        let model_path = self.resolve_path(&model_config.path);

        // Try custom factories first
        for factory in &self.custom_factories {
            if let Some(backend) = factory(&model_config, &self.model_root)? {
                tracing::info!(
                    model_id,
                    backend = backend.backend_name(),
                    "ModelRegistry: custom backend loaded"
                );
                models.insert(
                    model_id.to_string(),
                    Arc::new(LoadedModel { backend }),
                );
                return Ok(());
            }
        }

        // Built-in backends
        let backend: Box<dyn ModelBackend> = match model_config.backend.as_str() {
            "onnx-text" => {
                let labels = self.load_label_map_i64(&model_config)?;
                Box::new(OnnxTextBackend::new(
                    &model_path,
                    labels,
                    model_config.min_confidence,
                )?)
            }
            "onnx-embed" => {
                let labels = self.load_label_map_usize(&model_config)?;
                Box::new(OnnxEmbedBackend::new(
                    &model_path,
                    labels,
                    model_config.embedding_dim.unwrap_or(384),
                    model_config.min_confidence,
                )?)
            }
            other => {
                return Err(anyhow!(
                    "unsupported backend type '{}' for model '{}'. \
                     Register a custom BackendFactory for non-built-in backends.",
                    other,
                    model_id
                ));
            }
        };

        tracing::info!(
            model_id,
            backend = model_config.backend.as_str(),
            "ModelRegistry: model loaded"
        );

        models.insert(model_id.to_string(), Arc::new(LoadedModel { backend }));

        Ok(())
    }

    /// Hot-reload a model from a new path. The model must already be loaded.
    pub fn reload_model(&self, model_id: &str, new_path: &Path) -> anyhow::Result<()> {
        let models = self
            .models
            .read()
            .map_err(|_| anyhow!("models RwLock poisoned"))?;

        let loaded = models
            .get(model_id)
            .ok_or_else(|| anyhow!("model '{}' not loaded, can't reload", model_id))?;

        loaded.backend.reload(new_path)?;
        tracing::info!(model_id, path = %new_path.display(), "ModelRegistry: model hot-reloaded");
        Ok(())
    }

    /// Unload a model, freeing its memory.
    pub fn unload_model(&self, model_id: &str) -> anyhow::Result<bool> {
        let mut models = self
            .models
            .write()
            .map_err(|_| anyhow!("models RwLock poisoned"))?;
        let removed = models.remove(model_id).is_some();
        if removed {
            tracing::info!(model_id, "ModelRegistry: model unloaded");
        }
        Ok(removed)
    }

    /// Get the config for a model.
    pub fn get_model_config(&self, model_id: &str) -> Option<ModelConfig> {
        self.config
            .read()
            .ok()
            .and_then(|c| c.get_model(model_id).cloned())
    }

    /// Replace the entire config (e.g. on TOML hot-reload).
    ///
    /// Does NOT unload existing models — they stay loaded until explicitly
    /// unloaded or the registry is dropped.
    pub fn update_config(&self, config: MultimindConfig) {
        if let Ok(mut c) = self.config.write() {
            *c = config;
            tracing::info!("ModelRegistry: config updated");
        }
    }

    /// Register an already-constructed backend under the given model ID.
    ///
    /// Useful for programmatic registration without going through TOML config.
    pub fn register_model(
        &self,
        model_id: impl Into<String>,
        backend: Box<dyn ModelBackend>,
    ) -> anyhow::Result<()> {
        let id = model_id.into();
        let mut models = self
            .models
            .write()
            .map_err(|_| anyhow!("models RwLock poisoned"))?;
        tracing::info!(
            model_id = %id,
            backend = backend.backend_name(),
            "ModelRegistry: model registered programmatically"
        );
        models.insert(id, Arc::new(LoadedModel { backend }));
        Ok(())
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    fn resolve_path(&self, config_path: &str) -> PathBuf {
        let p = Path::new(config_path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.model_root.join(p)
        }
    }

    fn load_label_map_i64(
        &self,
        model_config: &ModelConfig,
    ) -> anyhow::Result<HashMap<i64, String>> {
        if let Some(ref labels_path) = model_config.labels {
            let full_path = self.resolve_path(labels_path);
            OnnxTextBackend::load_labels(&full_path).with_context(|| {
                format!(
                    "failed to load labels for model '{}'",
                    model_config.id
                )
            })
        } else {
            Ok(HashMap::new())
        }
    }

    fn load_label_map_usize(
        &self,
        model_config: &ModelConfig,
    ) -> anyhow::Result<HashMap<usize, String>> {
        if let Some(ref labels_path) = model_config.labels {
            let full_path = self.resolve_path(labels_path);
            OnnxEmbedBackend::load_labels(&full_path).with_context(|| {
                format!(
                    "failed to load labels for model '{}'",
                    model_config.id
                )
            })
        } else {
            Ok(HashMap::new())
        }
    }
}
