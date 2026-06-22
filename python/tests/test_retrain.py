"""Tests for retrain types — mirrors Rust retrain/types.rs tests."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from multimind.types import TrainingSignal
from multimind.retrain.types import (
    RetrainArtifact,
    RetrainConfig,
    SignalFeatures,
    extract_features,
    learn_weights,
)


# ── Test weight model ───────────────────────────────────────────────────────


class SampleWeightModel:
    """Simple weight model for testing."""

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
        new = SampleWeightModel(
            version=self._version,
            adjustments=dict(self._adjustments),
        )
        return new


# ── Feature extraction tests ───────────────────────────────────────────────


def test_extract_features_empty():
    features = extract_features([])
    assert features.total == 0
    assert len(features.category_signals) == 0


def test_extract_features_correct_predictions():
    signals = [
        TrainingSignal(
            model_id="test",
            input_text="hello",
            predicted_label="safe",
            corrected_label="safe",
            original_confidence=0.9,
        ),
        TrainingSignal(
            model_id="test",
            input_text="world",
            predicted_label="safe",
            corrected_label="safe",
            original_confidence=0.8,
        ),
    ]

    features = extract_features(signals)
    assert features.total == 2
    safe = features.category_signals["safe"]
    assert safe.total == 2
    assert safe.correct == 2
    assert safe.corrections == 0


def test_extract_features_with_corrections():
    signals = [
        TrainingSignal(
            model_id="test",
            input_text="pii data",
            predicted_label="safe",
            corrected_label="unsafe",
            original_confidence=0.7,
        ),
    ]

    features = extract_features(signals)
    assert features.total == 1
    safe = features.category_signals["safe"]
    assert safe.corrections == 1
    assert "unsafe" in features.category_signals


# ── Weight learning tests ───────────────────────────────────────────────────


def test_learn_weights_no_change_below_threshold():
    model = SampleWeightModel(
        version=0,
        adjustments={"safe": 1.0, "unsafe": 1.0},
    )

    # Only 2 corrections — below default min_corrections_for_update of 5
    signals = [
        TrainingSignal(
            model_id="test",
            input_text="a",
            predicted_label="safe",
            corrected_label="unsafe",
            original_confidence=0.6,
        ),
        TrainingSignal(
            model_id="test",
            input_text="b",
            predicted_label="safe",
            corrected_label="unsafe",
            original_confidence=0.5,
        ),
    ]

    features = extract_features(signals)
    config = RetrainConfig()
    updates = learn_weights(model, features, config)

    # Adjustments shouldn't change (below threshold)
    assert abs(updates["safe"] - 1.0) < 1e-9


# ── Artifact tests ──────────────────────────────────────────────────────────


def test_artifact_integrity():
    model = SampleWeightModel(
        version=1,
        adjustments={"a": 0.8, "b": 1.2},
    )

    artifact = RetrainArtifact.from_model(model, "test", 100)
    assert artifact.verify()
    assert artifact.version == 1
    assert artifact.signals_consumed == 100


def test_artifact_save_load():
    model = SampleWeightModel(
        version=3,
        adjustments={"cat_a": 0.9, "cat_b": 1.1},
    )

    artifact = RetrainArtifact.from_model(model, "test_model", 42)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = artifact.save(tmpdir)
        assert path.exists()

        # Load it back
        loaded = RetrainArtifact.load(path)
        assert loaded.model_id == "test_model"
        assert loaded.version == 3
        assert loaded.signals_consumed == 42
        assert loaded.verify()

        # Check latest pointer
        latest_path = Path(tmpdir) / "test_model_latest.json"
        assert latest_path.exists()
        latest = RetrainArtifact.load(latest_path)
        assert latest.version == 3


def test_artifact_tamper_detection():
    model = SampleWeightModel(
        version=1,
        adjustments={"x": 1.0},
    )

    artifact = RetrainArtifact.from_model(model, "test", 10)
    assert artifact.verify()

    # Tamper with the weight matrix
    artifact.weight_matrix[0] = 999.0
    assert not artifact.verify()
