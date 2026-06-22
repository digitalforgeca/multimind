"""TOML-based configuration for the model registry."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelRetrainConfig:
    """Per-model retrain thresholds.

    Attributes:
        min_signals: Minimum correction signals before retraining.
        min_sessions: Minimum classification sessions before retraining.
    """
    min_signals: int = 10
    min_sessions: int = 20


@dataclass
class ModelConfig:
    """Configuration for a single model.

    Attributes:
        id: Unique identifier (e.g. "sivu", "vibeguard").
        backend: Backend type: "onnx-text", "onnx-embed", or custom name.
        path: Path to the model file. Relative to model root.
        labels: Path to the label map JSON file. Optional.
        classes: Expected class names. Optional.
        min_confidence: Minimum confidence to accept a classification.
        embedding_dim: Number of embedding dimensions (onnx-embed only).
        retrain: Retrain configuration. Optional.
    """
    id: str
    backend: str
    path: str
    labels: str | None = None
    classes: list[str] | None = None
    min_confidence: float = 0.5
    embedding_dim: int | None = None
    retrain: ModelRetrainConfig | None = None


@dataclass
class MultimindConfig:
    """Top-level multimind configuration.

    Attributes:
        models: Registered models.
    """
    models: list[ModelConfig] = field(default_factory=list)

    @classmethod
    def from_toml(cls, toml_str: str) -> MultimindConfig:
        """Parse from a TOML string."""
        raw = tomllib.loads(toml_str)
        return cls._from_dict(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> MultimindConfig:
        """Parse from a TOML file path."""
        p = Path(path)
        with p.open("rb") as f:
            raw = tomllib.load(f)
        return cls._from_dict(raw)

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Find a model config by ID."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    @classmethod
    def _from_dict(cls, raw: dict) -> MultimindConfig:
        """Build config from parsed TOML dict."""
        models: list[ModelConfig] = []
        for m in raw.get("models", []):
            retrain_raw = m.get("retrain")
            retrain = None
            if retrain_raw is not None:
                retrain = ModelRetrainConfig(
                    min_signals=retrain_raw.get("min_signals", 10),
                    min_sessions=retrain_raw.get("min_sessions", 20),
                )
            models.append(ModelConfig(
                id=m["id"],
                backend=m["backend"],
                path=m["path"],
                labels=m.get("labels"),
                classes=m.get("classes"),
                min_confidence=m.get("min_confidence", 0.5),
                embedding_dim=m.get("embedding_dim"),
                retrain=retrain,
            ))
        return cls(models=models)
