"""Retrain pipeline orchestrator.

Manages the background retrain loop: threshold checking, signal consumption,
feature extraction, weight learning, artifact export, and hot-swap.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import TYPE_CHECKING

from ..types import SignalStore, TrainingSignal
from .types import (
    RetrainArtifact,
    RetrainConfig,
    RetrainResult,
    RetrainStatus,
    WeightModel,
    extract_features,
    learn_weights,
)

if TYPE_CHECKING:
    from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


class RetrainPipeline:
    """The retrain pipeline for a single model.

    Manages a background loop that checks for accumulated signals,
    runs the retrain cycle, and hot-swaps the model in the registry.

    Args:
        config: Retrain configuration.
        model_id: Model identifier.
        baseline: Initial weight model.
    """

    def __init__(
        self,
        config: RetrainConfig,
        model_id: str,
        baseline: WeightModel,
    ) -> None:
        self._config = config
        self._model_id = model_id
        self._current_model = copy.deepcopy(baseline)
        self._latest_artifact: RetrainArtifact | None = None
        self._latest_result: RetrainResult | None = None
        self._running = False
        self._lock = threading.RLock()
        self._trigger = threading.Event()
        self._stop = threading.Event()
        self._bg_thread: threading.Thread | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def current_model(self) -> WeightModel:
        """Get a copy of the current weight model."""
        with self._lock:
            return copy.deepcopy(self._current_model)

    def latest_artifact(self) -> RetrainArtifact | None:
        """Get the latest artifact (if any retrain has completed)."""
        with self._lock:
            return self._latest_artifact

    def status(self, unconsumed_signals: int) -> RetrainStatus:
        """Get the current pipeline status."""
        with self._lock:
            return RetrainStatus(
                model_version=self._current_model.version(),
                unconsumed_signals=unconsumed_signals,
                threshold_met=unconsumed_signals >= self._config.signal_threshold,
                running=self._running,
                last_result=self._latest_result,
            )

    def trigger(self) -> None:
        """Manually trigger a retrain cycle (non-blocking)."""
        self._trigger.set()

    def run_retrain(
        self,
        signal_store: SignalStore,
        registry: ModelRegistry | None = None,
    ) -> RetrainResult:
        """Run a single retrain cycle synchronously.

        Exports signals from the store, extracts features, learns new weights,
        creates an artifact, and optionally hot-swaps the model in the registry.

        Raises:
            RuntimeError: If no pending signals or retrain fails.
        """
        start = time.monotonic()

        with self._lock:
            previous_version = self._current_model.version()
            self._running = True

        try:
            # 1. Export pending signals (limited at the DB level)
            batch = signal_store.export_pending(
                self._model_id, limit=self._config.batch_size
            )
            if not batch:
                raise RuntimeError("no pending signals")

            logger.info(
                "retrain: starting cycle for %s (%d signals)",
                self._model_id, len(batch),
            )

            # 2. Extract features
            features = extract_features(batch)

            # 3. Learn updated weights
            with self._lock:
                current = copy.deepcopy(self._current_model)

            updated = learn_weights(current, features, self._config)

            # 4. Create artifact
            artifact = RetrainArtifact.from_model(updated, self._model_id, len(batch))

            # 5. Persist artifact
            artifact_path: str | None = None
            try:
                path = artifact.save(self._config.artifact_dir)
                artifact_path = str(path)
            except Exception as e:
                logger.warning(
                    "retrain: failed to persist artifact for %s: %s (continuing)",
                    self._model_id, e,
                )

            # 6. Mark exported signals as consumed (targeted, not blanket)
            consumed_ids = [s.signal_id for s in batch if s.signal_id is not None]
            try:
                signal_store.mark_consumed(self._model_id, consumed_ids)
            except Exception as e:
                logger.error(
                    "retrain: failed to mark signals consumed for %s: %s",
                    self._model_id, e,
                )

            # 7. Update current model
            with self._lock:
                self._current_model = updated
                self._latest_artifact = artifact

            # 8. Hot-swap in registry if provided
            if registry is not None and artifact_path is not None:
                try:
                    registry.reload_model(self._model_id, artifact_path)
                except Exception as e:
                    logger.warning(
                        "retrain: hot-swap failed for %s: %s",
                        self._model_id, e,
                    )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            result = RetrainResult(
                model_id=self._model_id,
                new_version=updated.version(),
                previous_version=previous_version,
                signals_consumed=len(batch),
                artifact_path=artifact_path,
                duration_ms=elapsed_ms,
            )

            with self._lock:
                self._latest_result = result

            logger.info(
                "retrain: cycle complete for %s (v%d → v%d, %d signals, %dms)",
                self._model_id,
                result.previous_version,
                result.new_version,
                result.signals_consumed,
                result.duration_ms,
            )

            return result

        finally:
            with self._lock:
                self._running = False

    def start_background(
        self,
        signal_store: SignalStore,
        registry: ModelRegistry | None = None,
    ) -> threading.Thread:
        """Start a background retrain loop.

        Runs on a daemon thread, checking for accumulated signals at the
        configured interval. Can also be triggered manually via ``trigger()``.

        Returns:
            The background thread handle.
        """
        self._stop.clear()
        self._trigger.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                # Wait for interval or trigger
                triggered = self._trigger.wait(timeout=self._config.check_interval)
                if self._stop.is_set():
                    break
                if triggered:
                    self._trigger.clear()
                    logger.info("retrain: manual trigger received for %s", self._model_id)

                # Check threshold
                try:
                    pending = signal_store.count_pending(self._model_id)
                except Exception as e:
                    logger.error(
                        "retrain: failed to count pending signals for %s: %s",
                        self._model_id, e,
                    )
                    continue

                if pending < self._config.signal_threshold:
                    continue

                # Run retrain
                try:
                    self.run_retrain(signal_store, registry)
                except Exception as e:
                    logger.error(
                        "retrain: background cycle failed for %s: %s",
                        self._model_id, e,
                    )

        thread = threading.Thread(
            target=_loop,
            name=f"multimind-retrain-{self._model_id}",
            daemon=True,
        )
        self._bg_thread = thread
        thread.start()
        return thread

    def stop_background(self) -> None:
        """Stop the background retrain loop."""
        self._stop.set()
        self._trigger.set()  # Wake the thread so it can exit
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5.0)
            self._bg_thread = None
