"""Core types for multimind — ModelInput, Verdict, TrainingSignal, and protocols."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Protocol, runtime_checkable


# ── Model input ─────────────────────────────────────────────────────────────


class InputKind(Enum):
    """Discriminator for ModelInput variants."""
    TEXT = auto()
    EMBEDDING = auto()
    STRUCTURED = auto()


@dataclass(frozen=True)
class ModelInput:
    """Input to a model. Backends accept one or more of these variants.

    Attributes:
        kind: Which variant this input represents.
        text: Raw text (for TF-IDF ONNX models). Set when kind=TEXT.
        embedding: Pre-computed embedding vector. Set when kind=EMBEDDING.
        structured: Arbitrary JSON-serialisable dict. Set when kind=STRUCTURED.
    """
    kind: InputKind
    text: str | None = None
    embedding: list[float] | None = None
    structured: dict | None = None

    @classmethod
    def from_text(cls, text: str) -> ModelInput:
        """Create a text input."""
        return cls(kind=InputKind.TEXT, text=text)

    @classmethod
    def from_embedding(cls, embedding: list[float]) -> ModelInput:
        """Create an embedding input."""
        return cls(kind=InputKind.EMBEDDING, embedding=embedding)

    @classmethod
    def from_structured(cls, data: dict) -> ModelInput:
        """Create a structured (JSON) input."""
        return cls(kind=InputKind.STRUCTURED, structured=data)


# ── Verdict ─────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """The output of a single model inference call.

    Attributes:
        label: Winning label (e.g. "store", "episodic", "SAFE").
        confidence: Confidence in the winning label (0.0 – 1.0).
        all_scores: Per-class scores (label → probability).
            Empty dict if the backend doesn't support per-class output.
    """
    label: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)


# ── Training signal ─────────────────────────────────────────────────────────


@dataclass
class TrainingSignal:
    """A correction signal for model improvement.

    The consuming service records these; the retrain pipeline reads them.

    Attributes:
        model_id: Which model produced the original verdict.
        input_text: The input that was classified.
        predicted_label: The model's original prediction.
        corrected_label: The corrected label (ground truth from user/system).
        original_confidence: Optional confidence of the original prediction.
        signal_id: Storage-assigned row ID (populated on export, ``None`` when creating).
    """
    model_id: str
    input_text: str
    predicted_label: str
    corrected_label: str
    original_confidence: float | None = None
    signal_id: str | None = None

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        d = {
            "model_id": self.model_id,
            "input_text": self.input_text,
            "predicted_label": self.predicted_label,
            "corrected_label": self.corrected_label,
            "original_confidence": self.original_confidence,
        }
        if self.signal_id is not None:
            d["signal_id"] = self.signal_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainingSignal:
        """Deserialise from a plain dict."""
        return cls(
            model_id=d["model_id"],
            input_text=d["input_text"],
            predicted_label=d["predicted_label"],
            corrected_label=d["corrected_label"],
            original_confidence=d.get("original_confidence"),
            signal_id=d.get("signal_id"),
        )


# ── Protocols (trait equivalents) ───────────────────────────────────────────


@runtime_checkable
class ModelBackend(Protocol):
    """A model backend that can classify inputs.

    Implement this protocol for custom inference engines.
    """

    def classify(self, input: ModelInput) -> Verdict:
        """Run inference on the given input."""
        ...

    def reload(self, path: Path) -> None:
        """Hot-reload the model from a new path.

        Raises on failure (old model stays loaded).
        """
        ...

    def backend_name(self) -> str:
        """Human-readable backend name (e.g. 'onnx-text', 'onnx-embed')."""
        ...


@runtime_checkable
class SignalStore(Protocol):
    """Storage backend for training signals.

    Implement this protocol for custom storage (Redis, JSONL, etc.).
    """

    def record(self, signal: TrainingSignal) -> None:
        """Record a correction signal."""
        ...

    def count_pending(self, model_id: str) -> int:
        """Count signals for a given model since last retrain."""
        ...

    def export_pending(
        self, model_id: str, *, limit: int | None = None
    ) -> list[TrainingSignal]:
        """Export pending signals for retraining.

        Args:
            model_id: Model to export signals for.
            limit: Maximum rows to return. ``None`` means all pending.
        """
        ...

    def mark_consumed(self, model_id: str, signal_ids: list[str]) -> None:
        """Mark specific signals as consumed (after successful retrain).

        Only the signals with the given IDs are marked. This prevents
        silently eating signals that arrived between export and consume.
        """
        ...

    def mark_all_consumed(self, model_id: str) -> None:
        """Mark **all** pending signals for a model as consumed.

        Use sparingly — prefer ``mark_consumed`` with specific IDs to
        avoid racing with newly-arrived signals.
        """
        ...
