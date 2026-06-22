//! Retrain pipeline orchestrator.
//!
//! Manages the background retrain loop: threshold checking, signal consumption,
//! feature extraction, weight learning, artifact export, and hot-swap.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tokio::sync::Notify;
use tracing::{error, info, warn};

use crate::registry::ModelRegistry;
use crate::SignalStore;

use super::types::*;

/// The retrain pipeline for a single model.
///
/// Manages a background loop that checks for accumulated signals,
/// runs the retrain cycle, and hot-swaps the model in the registry.
///
/// Generic over `M: WeightModel` — consumers define their domain-specific
/// weight model shape.
pub struct RetrainPipeline<M: WeightModel> {
    config: RetrainConfig,
    model_id: String,
    current_model: Arc<RwLock<M>>,
    latest_artifact: Arc<RwLock<Option<RetrainArtifact>>>,
    latest_result: Arc<RwLock<Option<RetrainResult>>>,
    running: Arc<RwLock<bool>>,
    trigger: Arc<Notify>,
}

impl<M: WeightModel + 'static> RetrainPipeline<M> {
    /// Create a new retrain pipeline.
    pub fn new(config: RetrainConfig, model_id: impl Into<String>, baseline: M) -> Self {
        Self {
            config,
            model_id: model_id.into(),
            current_model: Arc::new(RwLock::new(baseline)),
            latest_artifact: Arc::new(RwLock::new(None)),
            latest_result: Arc::new(RwLock::new(None)),
            running: Arc::new(RwLock::new(false)),
            trigger: Arc::new(Notify::new()),
        }
    }

    /// Get the current weight model.
    pub fn current_model(&self) -> M {
        self.current_model.read().clone()
    }

    /// Get the latest artifact (if any retrain has completed).
    pub fn latest_artifact(&self) -> Option<RetrainArtifact> {
        self.latest_artifact.read().clone()
    }

    /// Get the current status.
    pub fn status(&self, unconsumed_signals: usize) -> RetrainStatus {
        RetrainStatus {
            model_version: self.current_model.read().version(),
            unconsumed_signals,
            threshold_met: unconsumed_signals >= self.config.signal_threshold,
            running: *self.running.read(),
            last_result: self.latest_result.read().clone(),
        }
    }

    /// Whether a retrain cycle is currently running.
    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

    /// Manually trigger a retrain cycle (non-blocking).
    pub fn trigger(&self) {
        self.trigger.notify_one();
    }

    /// Run a single retrain cycle synchronously.
    ///
    /// Exports signals from the store (up to `batch_size`), extracts features,
    /// learns new weights, creates an artifact, and optionally hot-swaps the
    /// model in the registry.
    pub fn run_retrain(
        &self,
        signal_store: &dyn SignalStore,
        registry: Option<&ModelRegistry>,
    ) -> Result<RetrainResult, String> {
        let start = Instant::now();
        let previous_version = self.current_model.read().version();

        // Mark as running (scopeguard resets on any exit path)
        *self.running.write() = true;
        let running_guard = self.running.clone();
        let _guard = scopeguard::guard((), move |_| {
            *running_guard.write() = false;
        });

        // 1. Export pending signals (limited to batch_size at the DB level)
        let batch = signal_store
            .export_pending(&self.model_id, Some(self.config.batch_size))
            .map_err(|e| format!("failed to export signals: {e}"))?;

        if batch.is_empty() {
            return Err("no pending signals".into());
        }

        info!(
            model_id = %self.model_id,
            signals = batch.len(),
            "retrain: starting cycle"
        );

        // 2. Extract features
        let features = extract_features(&batch);

        // 3. Learn updated weights
        let current = self.current_model.read().clone();
        let updated = learn_weights(&current, &features, &self.config);

        // 4. Create artifact
        let artifact = RetrainArtifact::from_model(&updated, &self.model_id, batch.len());

        // 5. Persist artifact
        let artifact_path = match artifact.save(&self.config.artifact_dir) {
            Ok(path) => Some(path.to_string_lossy().to_string()),
            Err(e) => {
                warn!(
                    model_id = %self.model_id,
                    error = %e,
                    "retrain: failed to persist artifact (continuing)"
                );
                None
            }
        };

        // 6. Mark exported signals as consumed (targeted, not blanket)
        let consumed_ids: Vec<String> = batch
            .iter()
            .filter_map(|s| s.signal_id.clone())
            .collect();
        if let Err(e) = signal_store.mark_consumed(&self.model_id, &consumed_ids) {
            error!(
                model_id = %self.model_id,
                error = %e,
                "retrain: failed to mark signals consumed"
            );
        }

        // 7. Update current model + artifact
        *self.current_model.write() = updated;
        *self.latest_artifact.write() = Some(artifact);

        // 8. Hot-swap in registry if provided
        if let (Some(registry), Some(ref path)) = (registry, &artifact_path) {
            if let Err(e) = registry.reload_model(&self.model_id, std::path::Path::new(path)) {
                warn!(
                    model_id = %self.model_id,
                    error = %e,
                    "retrain: hot-swap failed (model will use new weights on next restart)"
                );
            }
        }

        let result = RetrainResult {
            model_id: self.model_id.clone(),
            new_version: self.current_model.read().version(),
            previous_version,
            signals_consumed: batch.len(),
            artifact_path,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        *self.latest_result.write() = Some(result.clone());

        info!(
            model_id = %self.model_id,
            version = result.new_version,
            signals = result.signals_consumed,
            duration_ms = result.duration_ms,
            "retrain: cycle complete"
        );

        Ok(result)
    }

    /// Start a background retrain loop.
    ///
    /// Runs on a tokio task, checking for threshold signals at the configured
    /// interval. Can also be triggered manually via [`trigger()`](Self::trigger).
    ///
    /// Returns a handle that can be used to abort the background task.
    pub fn start_background(
        self: &Arc<Self>,
        signal_store: Arc<dyn SignalStore>,
        registry: Option<Arc<ModelRegistry>>,
    ) -> tokio::task::JoinHandle<()>
    where
        M: 'static,
    {
        let pipeline = Arc::clone(self);
        let trigger = self.trigger.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(pipeline.config.check_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = interval.tick() => {},
                    _ = trigger.notified() => {
                        info!(model_id = %pipeline.model_id, "retrain: manual trigger received");
                    },
                }

                // Check threshold before running (avoids loading signals unnecessarily)
                let pending = match signal_store.count_pending(&pipeline.model_id) {
                    Ok(n) => n,
                    Err(e) => {
                        error!(
                            model_id = %pipeline.model_id,
                            error = %e,
                            "retrain: failed to count pending signals"
                        );
                        continue;
                    }
                };

                if pending < pipeline.config.signal_threshold {
                    continue;
                }

                // Delegate to run_retrain — single implementation, no duplication
                let reg_ref = registry.as_deref();
                match pipeline.run_retrain(signal_store.as_ref(), reg_ref) {
                    Ok(result) => {
                        info!(
                            model_id = %result.model_id,
                            version = result.new_version,
                            signals = result.signals_consumed,
                            duration_ms = result.duration_ms,
                            "retrain: background cycle complete"
                        );
                    }
                    Err(e) => {
                        warn!(
                            model_id = %pipeline.model_id,
                            error = %e,
                            "retrain: background cycle failed"
                        );
                    }
                }
            }
        })
    }
}
