"""multimind — Multi-Model Mind Python SDK.

A generic ONNX model registry with inference, correction signals,
and a retrain pipeline for Python applications.

Multimind has **zero knowledge** of any particular product, domain,
or storage layer. Wire it into your own routing, storage, and
deployment systems.
"""

from .types import (
    InputKind,
    ModelInput,
    Verdict,
    TrainingSignal,
    ModelBackend,
    SignalStore,
)
from .config import MultimindConfig, ModelConfig, ModelRetrainConfig
from .registry import ModelRegistry

__version__ = "0.1.0"

__all__ = [
    # Core types
    "InputKind",
    "ModelInput",
    "Verdict",
    "TrainingSignal",
    # Protocols
    "ModelBackend",
    "SignalStore",
    # Config
    "MultimindConfig",
    "ModelConfig",
    "ModelRetrainConfig",
    # Registry
    "ModelRegistry",
]
