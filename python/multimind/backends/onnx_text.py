"""ONNX backend for string-input models (TF-IDF + SGD pipeline).

These models accept raw text as a string tensor — no pre-embedding needed.
Supports two sklearn ONNX export formats:

- **Format A** (Pipeline export): string label + sequence of maps (probability dict)
- **Format B** (OVR export): i64 label index + f32/f64 probability tensor
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..types import ModelInput, InputKind, Verdict

logger = logging.getLogger(__name__)


class OnnxTextBackend:
    """ONNX backend for TF-IDF text classification models.

    Expects models exported from scikit-learn with a string input tensor.
    Auto-detects the output format (string labels vs i64 indices).

    Args:
        model_path: Path to the ONNX model file.
        labels: Map of integer index → label name.
        min_confidence: Minimum confidence threshold.
    """

    def __init__(
        self,
        model_path: Path,
        labels: dict[int, str] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self._labels = labels or {}
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._session = ort.InferenceSession(str(model_path))
        logger.info("OnnxTextBackend: model loaded from %s", model_path)

    @staticmethod
    def load_labels(path: Path) -> dict[int, str]:
        """Load labels from a JSON file: ``{"0": "label_a", "1": "label_b", ...}``"""
        with open(path) as f:
            raw: dict[str, str] = json.load(f)
        return {int(k): v for k, v in raw.items()}

    @property
    def min_confidence(self) -> float:
        """Minimum confidence threshold for this backend."""
        return self._min_confidence

    def classify(self, input: ModelInput) -> Verdict:
        """Run inference on a text input.

        Raises:
            ValueError: If input is not ModelInput.TEXT.
            RuntimeError: If inference fails.
        """
        if input.kind != InputKind.TEXT or input.text is None:
            raise ValueError("OnnxTextBackend requires ModelInput.from_text()")

        label, all_scores = self._run_inference(input.text)
        confidence = all_scores.get(label, 0.0)
        return Verdict(label=label, confidence=confidence, all_scores=all_scores)

    def reload(self, path: Path) -> None:
        """Hot-reload the model from a new ONNX file."""
        new_session = ort.InferenceSession(str(path))
        with self._lock:
            self._session = new_session
        logger.info("OnnxTextBackend: model hot-reloaded from %s", path)

    def backend_name(self) -> str:
        return "onnx-text"

    # ── Internal ────────────────────────────────────────────────────────────

    def _run_inference(self, text: str) -> tuple[str, dict[str, float]]:
        """Run string-input ONNX inference and return (label, per-class probabilities)."""
        input_array = np.array([[text]], dtype=object)

        with self._lock:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_array})

        # ── Try Format A: string labels ──
        label_output = outputs[0]
        if isinstance(label_output, np.ndarray) and label_output.dtype.kind in ("U", "S", "O"):
            label = str(label_output.flat[0])
            all_scores: dict[str, float] = {}

            if len(outputs) >= 2:
                prob_output = outputs[1]
                # sklearn ZipMap produces a list of dicts
                if isinstance(prob_output, list) and len(prob_output) > 0 and isinstance(prob_output[0], dict):
                    all_scores = {str(k): float(v) for k, v in prob_output[0].items()}
                elif isinstance(prob_output, np.ndarray):
                    # Probability matrix — use labels if available
                    probs = prob_output.flatten()
                    for idx, p in enumerate(probs):
                        name = self._labels.get(idx, f"class_{idx}")
                        all_scores[name] = float(p)

            if not all_scores:
                all_scores = {label: 1.0}

            return label, all_scores

        # ── Try Format B: integer label index ──
        if isinstance(label_output, np.ndarray) and label_output.dtype.kind in ("i", "u"):
            label_idx = int(label_output.flat[0])
            label = self._labels.get(label_idx, f"class_{label_idx}")

            all_scores = {}
            if len(outputs) >= 2:
                prob_output = outputs[1]
                if isinstance(prob_output, np.ndarray):
                    probs = prob_output.flatten().astype(np.float32)
                    for idx, name in self._labels.items():
                        if 0 <= idx < len(probs):
                            all_scores[name] = float(probs[idx])
                elif isinstance(prob_output, list) and len(prob_output) > 0 and isinstance(prob_output[0], dict):
                    all_scores = {str(k): float(v) for k, v in prob_output[0].items()}

            if not all_scores:
                all_scores = {label: 1.0}

            return label, all_scores

        raise RuntimeError("failed to extract label tensor (neither string nor integer)")
