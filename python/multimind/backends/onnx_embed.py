"""ONNX backend for embedding-input models (pre-computed vector → classification).

These models accept a float32 embedding vector as input.
Suitable for sentence-transformer embeddings (384-dim, 768-dim, etc.)
fed into lightweight classifiers (LogReg, SVM, small MLP).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..types import ModelInput, InputKind, Verdict

logger = logging.getLogger(__name__)


def _softmax(logits: list[float]) -> list[float]:
    """Compute softmax over a logit list."""
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class OnnxEmbedBackend:
    """ONNX backend for embedding-based classifiers.

    Expects models that take a float32 tensor of shape ``[1, embedding_dim]``.

    Args:
        model_path: Path to the ONNX model file.
        labels: Map of integer index → label name.
        embedding_dim: Expected embedding dimension.
        min_confidence: Minimum confidence threshold.
    """

    def __init__(
        self,
        model_path: Path,
        labels: dict[int, str] | None = None,
        embedding_dim: int = 384,
        min_confidence: float = 0.5,
    ) -> None:
        self._labels = labels or {}
        self._embedding_dim = embedding_dim
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._session = ort.InferenceSession(str(model_path))
        logger.info(
            "OnnxEmbedBackend: model loaded from %s (dim=%d)", model_path, embedding_dim,
        )

    @staticmethod
    def load_labels(path: Path) -> dict[int, str]:
        """Load labels from a JSON file: ``{"0": "label_a", "1": "label_b", ...}``"""
        with open(path) as f:
            raw: dict[str, str] = json.load(f)
        return {int(k): v for k, v in raw.items()}

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension expected by this backend."""
        return self._embedding_dim

    def classify(self, input: ModelInput) -> Verdict:
        """Run inference on an embedding input.

        Raises:
            ValueError: If input is not ModelInput.EMBEDDING or wrong dimension.
            RuntimeError: If inference fails.
        """
        if input.kind != InputKind.EMBEDDING or input.embedding is None:
            raise ValueError("OnnxEmbedBackend requires ModelInput.from_embedding()")

        if len(input.embedding) != self._embedding_dim:
            raise ValueError(
                f"expected embedding of length {self._embedding_dim}, "
                f"got {len(input.embedding)}"
            )

        input_array = np.array([input.embedding], dtype=np.float32)

        with self._lock:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_array})

        # Extract raw output — may be logits or probabilities
        if len(outputs) >= 2:
            prob_output = outputs[1]
            if isinstance(prob_output, np.ndarray):
                raw = prob_output.flatten().astype(np.float32).tolist()[:len(self._labels) or None]
            elif isinstance(prob_output, list) and len(prob_output) > 0 and isinstance(prob_output[0], dict):
                # sklearn ZipMap format
                raw = [float(prob_output[0].get(self._labels.get(i, f"class_{i}"), 0.0))
                       for i in range(len(self._labels))]
            else:
                raw = outputs[0].flatten().astype(np.float32).tolist()[:len(self._labels) or None]
        else:
            raw = outputs[0].flatten().astype(np.float32).tolist()[:len(self._labels) or None]

        # Apply softmax if values aren't already probabilities
        is_prob = all(0.0 <= v <= 1.0 for v in raw) and abs(sum(raw) - 1.0) < 0.1
        probs = raw if is_prob else _softmax(raw)

        # Find winning label
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_conf = probs[best_idx]
        label = self._labels.get(best_idx, f"class_{best_idx}")

        all_scores: dict[str, float] = {}
        for i, p in enumerate(probs):
            name = self._labels.get(i, f"class_{i}")
            all_scores[name] = p

        return Verdict(label=label, confidence=best_conf, all_scores=all_scores)

    def reload(self, path: Path) -> None:
        """Hot-reload the model from a new ONNX file."""
        new_session = ort.InferenceSession(str(path))
        with self._lock:
            self._session = new_session
        logger.info("OnnxEmbedBackend: model hot-reloaded from %s", path)

    def backend_name(self) -> str:
        return "onnx-embed"
