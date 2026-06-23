"""Retrain pipeline types — protocols, dataclasses, and feature extraction."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import TrainingSignal

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class RetrainConfig:
    """Retrain pipeline configuration.

    Attributes:
        signal_threshold: Minimum unconsumed signals before a retrain is eligible.
        batch_size: Maximum signals to consume per retrain batch.
        check_interval: Background check interval in seconds.
        learning_rate: Learning rate for weight updates (0.0–1.0).
        min_corrections_for_update: Minimum correction signals for a category
            before applying its update.
        artifact_dir: Directory for persisting model artifacts.
    """
    signal_threshold: int = 200
    batch_size: int = 1000
    check_interval: float = 3600.0
    learning_rate: float = 0.05
    min_corrections_for_update: int = 5
    artifact_dir: str = "/tmp/multimind-models"


# ── Weight model protocol ──────────────────────────────────────────────────


@runtime_checkable
class WeightModel(Protocol):
    """A learnable weight model.

    Consumers implement this for their domain-specific model shape.
    The retrain pipeline operates on this protocol generically.

    Must be JSON-serialisable (for artifact export) and copyable.
    """

    def version(self) -> int:
        """Model version (monotonically increasing)."""
        ...

    def set_version(self, version: int) -> None:
        """Set the model version."""
        ...

    def categories(self) -> list[str]:
        """All category names this model tracks."""
        ...

    def adjustment(self, category: str) -> float:
        """Get the weight adjustment for a category (1.0 = no change)."""
        ...

    def set_adjustment(self, category: str, value: float) -> None:
        """Set the weight adjustment for a category."""
        ...


# ── Feature extraction ──────────────────────────────────────────────────────


@dataclass
class CategoryFeatures:
    """Features for a single category.

    Attributes:
        total: Total signals where this category was involved.
        correct: Signals where the model prediction was correct.
        corrections: Signals where the model prediction was wrong.
        avg_confidence_correct: Average confidence on correct predictions.
        avg_confidence_incorrect: Average confidence on incorrect predictions.
    """
    total: int = 0
    correct: int = 0
    corrections: int = 0
    avg_confidence_correct: float = 0.0
    avg_confidence_incorrect: float = 0.0


@dataclass
class SignalFeatures:
    """Extracted features from a batch of training signals.

    Attributes:
        total: Total signals in the batch.
        category_signals: Per-category signal counts and correction rates.
    """
    total: int = 0
    category_signals: dict[str, CategoryFeatures] = field(default_factory=dict)


def extract_features(signals: list[TrainingSignal]) -> SignalFeatures:
    """Extract features from a batch of training signals.

    Domain-agnostic: groups by predicted/corrected labels and computes
    correction rates and confidence distributions.
    """
    # (total, correct, corrections, sum_conf_correct, sum_conf_incorrect)
    category_map: dict[str, list[float]] = {}

    def get_entry(cat: str) -> list[float]:
        if cat not in category_map:
            category_map[cat] = [0.0, 0.0, 0.0, 0.0, 0.0]
        return category_map[cat]

    for signal in signals:
        is_correct = signal.predicted_label == signal.corrected_label
        confidence = signal.original_confidence if signal.original_confidence is not None else 0.5

        # Count for predicted category
        entry = get_entry(signal.predicted_label)
        entry[0] += 1  # total
        if is_correct:
            entry[1] += 1  # correct
            entry[3] += confidence  # sum correct confidence
        else:
            entry[2] += 1  # corrections
            entry[4] += confidence  # sum incorrect confidence

        # Also count for corrected category (if different)
        if not is_correct:
            corrected_entry = get_entry(signal.corrected_label)
            corrected_entry[0] += 1
            corrected_entry[1] += 1  # The correction itself is "correct" for the target
            corrected_entry[3] += confidence

    category_signals: dict[str, CategoryFeatures] = {}
    for cat, (total, correct, corrections, sum_conf_correct, sum_conf_incorrect) in category_map.items():
        total_i = int(total)
        correct_i = int(correct)
        corrections_i = int(corrections)
        category_signals[cat] = CategoryFeatures(
            total=total_i,
            correct=correct_i,
            corrections=corrections_i,
            avg_confidence_correct=sum_conf_correct / correct_i if correct_i > 0 else 0.0,
            avg_confidence_incorrect=sum_conf_incorrect / corrections_i if corrections_i > 0 else 0.0,
        )

    return SignalFeatures(total=len(signals), category_signals=category_signals)


def learn_weights(
    model: WeightModel,
    features: SignalFeatures,
    config: RetrainConfig,
) -> WeightModel:
    """Apply learned weight updates based on extracted features.

    Returns a deep copy of the model with updated version and adjustments.
    Mirrors the Rust implementation exactly.

    Uses the correction rate to adjust category weights:
    - High correction rate → suppress (lower weight)
    - Low correction rate → boost (higher weight)
    """
    import copy
    updated = copy.deepcopy(model)
    updated.set_version(model.version() + 1)

    for category in model.categories():
        cat_features = features.category_signals.get(category)
        if cat_features is None:
            continue

        # Skip if not enough corrections to be statistically meaningful
        if cat_features.corrections < config.min_corrections_for_update:
            continue

        correction_rate = (
            cat_features.corrections / cat_features.total
            if cat_features.total > 0
            else 0.0
        )

        # Adjustment: high correction rate → reduce weight, low → increase
        current = model.adjustment(category)
        delta = config.learning_rate * (1.0 - 2.0 * correction_rate)
        new_adjustment = max(0.1, min(10.0, current + delta))
        updated.set_adjustment(category, new_adjustment)

    return updated


# ── Artifact ────────────────────────────────────────────────────────────────


@dataclass
class RetrainArtifact:
    """A retrain artifact — the output of a successful retrain cycle.

    Attributes:
        model_id: Model ID this artifact is for.
        version: Model version.
        categories: Category names (row/column order for the weight matrix).
        weight_matrix: Flattened NxN diagonal weight matrix (row-major).
        checksum: SHA-256 checksum of the weight matrix bytes.
        created_at: When this artifact was created (ISO format string).
        signals_consumed: Number of signals consumed to produce this artifact.
    """
    model_id: str
    version: int
    categories: list[str]
    weight_matrix: list[float]
    checksum: str
    created_at: str
    signals_consumed: int

    @classmethod
    def from_model(
        cls,
        model: WeightModel,
        model_id: str,
        signals_consumed: int,
    ) -> RetrainArtifact:
        """Create an artifact from a weight model."""
        categories = model.categories()
        n = len(categories)

        # Build diagonal weight matrix
        weight_matrix = [0.0] * (n * n)
        for i, cat in enumerate(categories):
            weight_matrix[i * n + i] = model.adjustment(cat)

        # Compute checksum (same as Rust: f32 LE bytes → SHA-256)
        import struct
        matrix_bytes = b"".join(struct.pack("<f", w) for w in weight_matrix)
        checksum = hashlib.sha256(matrix_bytes).hexdigest()

        return cls(
            model_id=model_id,
            version=model.version(),
            categories=categories,
            weight_matrix=weight_matrix,
            checksum=checksum,
            created_at=datetime.now(timezone.utc).isoformat(),
            signals_consumed=signals_consumed,
        )

    def verify(self) -> bool:
        """Verify the integrity of this artifact."""
        import struct
        matrix_bytes = b"".join(struct.pack("<f", w) for w in self.weight_matrix)
        computed = hashlib.sha256(matrix_bytes).hexdigest()
        return computed == self.checksum

    def save(self, directory: str | None = None) -> Path:
        """Persist the artifact to disk as JSON.

        Args:
            directory: Directory to save in. Uses artifact_dir if not specified.

        Returns:
            Path to the saved versioned file.
        """
        dir_path = Path(directory) if directory else Path(self.model_id)
        dir_path.mkdir(parents=True, exist_ok=True)

        data = {
            "model_id": self.model_id,
            "version": self.version,
            "categories": self.categories,
            "weight_matrix": self.weight_matrix,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "signals_consumed": self.signals_consumed,
        }

        filename = f"{self.model_id}_v{self.version}.json"
        path = dir_path / filename
        path.write_text(json.dumps(data, indent=2))

        # Also write a "latest" pointer
        latest_path = dir_path / f"{self.model_id}_latest.json"
        latest_path.write_text(json.dumps(data, indent=2))

        logger.info("RetrainArtifact: saved %s v%d to %s", self.model_id, self.version, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RetrainArtifact:
        """Load an artifact from a JSON file.

        Raises:
            ValueError: If integrity check fails.
        """
        p = Path(path)
        data = json.loads(p.read_text())
        artifact = cls(
            model_id=data["model_id"],
            version=data["version"],
            categories=data["categories"],
            weight_matrix=data["weight_matrix"],
            checksum=data["checksum"],
            created_at=data["created_at"],
            signals_consumed=data["signals_consumed"],
        )
        if not artifact.verify():
            raise ValueError(
                f"artifact integrity check failed for {artifact.model_id} v{artifact.version}"
            )
        return artifact


# ── Retrain result / status ─────────────────────────────────────────────────


@dataclass
class RetrainResult:
    """Result of a retrain cycle.

    Attributes:
        model_id: Model ID that was retrained.
        new_version: New model version.
        previous_version: Previous model version.
        signals_consumed: Number of signals consumed.
        artifact_path: Path to the saved artifact (if persisted).
        duration_ms: Duration of the retrain cycle in milliseconds.
    """
    model_id: str
    new_version: int
    previous_version: int
    signals_consumed: int
    artifact_path: str | None = None
    duration_ms: int = 0


@dataclass
class RetrainStatus:
    """Current status of the retrain pipeline for a model.

    Attributes:
        model_version: Current model version.
        unconsumed_signals: Number of unconsumed signals.
        threshold_met: Whether the signal threshold is met for retraining.
        running: Whether a retrain is currently running.
        last_result: Last retrain result (if any).
    """
    model_version: int
    unconsumed_signals: int
    threshold_met: bool
    running: bool
    last_result: RetrainResult | None = None
