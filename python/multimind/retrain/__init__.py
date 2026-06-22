"""Generic retrain pipeline.

Consumes correction signals, extracts features, learns weight adjustments,
exports artifacts, and hot-swaps live models.

Architecture::

    SignalStore (pending signals)
      │ export_pending(model_id)
      ▼
    FeatureExtractor → SignalFeatures
      │
      ▼
    WeightLearner → WeightModel (updated adjustments)
      │
      ▼
    ArtifactExporter → RetrainArtifact
      │
      ▼
    ModelRegistry.reload_model() → hot-swap
"""

from .types import (
    RetrainConfig,
    WeightModel,
    SignalFeatures,
    CategoryFeatures,
    extract_features,
    learn_weights,
    RetrainArtifact,
    RetrainResult,
    RetrainStatus,
)
from .pipeline import RetrainPipeline

__all__ = [
    "RetrainConfig",
    "WeightModel",
    "SignalFeatures",
    "CategoryFeatures",
    "extract_features",
    "learn_weights",
    "RetrainArtifact",
    "RetrainResult",
    "RetrainStatus",
    "RetrainPipeline",
]
