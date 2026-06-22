"""Tests for the retrain pipeline — integration tests with SQLite store."""

from __future__ import annotations

import copy

from multimind.types import TrainingSignal
from multimind.signals.sqlite import SqliteSignalStore
from multimind.retrain.pipeline import RetrainPipeline
from multimind.retrain.types import RetrainConfig


class SimpleWeightModel:
    """Test weight model for pipeline integration tests."""

    def __init__(
        self,
        version: int = 0,
        adjustments: dict[str, float] | None = None,
    ) -> None:
        self._version = version
        self._adjustments = dict(adjustments) if adjustments else {}

    def version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = v

    def categories(self) -> list[str]:
        return list(self._adjustments.keys())

    def adjustment(self, category: str) -> float:
        return self._adjustments.get(category, 1.0)

    def set_adjustment(self, category: str, value: float) -> None:
        self._adjustments[category] = value

    def __deepcopy__(self, memo):
        return SimpleWeightModel(
            version=self._version,
            adjustments=dict(self._adjustments),
        )


def test_pipeline_run_retrain():
    """Test a full retrain cycle: record signals → run retrain → verify."""
    store = SqliteSignalStore.in_memory()

    # Record enough signals to meet threshold
    config = RetrainConfig(
        signal_threshold=5,
        min_corrections_for_update=2,
        learning_rate=0.1,
    )

    baseline = SimpleWeightModel(
        version=0,
        adjustments={"safe": 1.0, "unsafe": 1.0},
    )

    pipeline = RetrainPipeline(config, "test_model", baseline)

    # Record 10 signals with corrections
    for i in range(10):
        store.record(TrainingSignal(
            model_id="test_model",
            input_text=f"input {i}",
            predicted_label="safe",
            corrected_label="unsafe",
            original_confidence=0.6,
        ))

    # Verify we have pending signals
    assert store.count_pending("test_model") == 10

    # Run retrain
    result = pipeline.run_retrain(store)
    assert result.model_id == "test_model"
    assert result.new_version == 1
    assert result.previous_version == 0
    assert result.signals_consumed == 10

    # Signals should be consumed
    assert store.count_pending("test_model") == 0

    # Model should be updated
    current = pipeline.current_model()
    assert current.version() == 1

    # Status check
    status = pipeline.status(0)
    assert status.model_version == 1
    assert not status.threshold_met
    assert not status.running
    assert status.last_result is not None


def test_pipeline_no_signals_raises():
    """Test that run_retrain raises when there are no pending signals."""
    store = SqliteSignalStore.in_memory()
    config = RetrainConfig(signal_threshold=5)
    baseline = SimpleWeightModel(version=0, adjustments={"a": 1.0})
    pipeline = RetrainPipeline(config, "test", baseline)

    try:
        pipeline.run_retrain(store)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "no pending signals" in str(e)


def test_pipeline_status():
    """Test pipeline status reporting."""
    config = RetrainConfig(signal_threshold=100)
    baseline = SimpleWeightModel(version=5, adjustments={"cat": 1.0})
    pipeline = RetrainPipeline(config, "test", baseline)

    status = pipeline.status(50)
    assert status.model_version == 5
    assert status.unconsumed_signals == 50
    assert not status.threshold_met
    assert not status.running

    status = pipeline.status(200)
    assert status.threshold_met
