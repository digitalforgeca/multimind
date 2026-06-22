"""Model registry — load, manage, and query models by ID.

Models are loaded lazily on first ``classify()`` call and cached.
Hot-reload replaces the model in-place without dropping the registry.
Custom backends can be registered alongside the built-in ONNX backends.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from .config import ModelConfig, MultimindConfig
from .types import ModelBackend, ModelInput, Verdict

logger = logging.getLogger(__name__)

# Factory signature: (model_config, model_root) -> backend or None
BackendFactory = Callable[[ModelConfig, Path], ModelBackend | None]


class ModelRegistry:
    """The model registry. Thread-safe, supports lazy loading and hot-reload.

    Args:
        config: The multimind configuration.
        model_root: Root directory for resolving relative model paths.
    """

    def __init__(self, config: MultimindConfig, model_root: str | Path = ".") -> None:
        self._config = config
        self._model_root = Path(model_root)
        self._models: dict[str, ModelBackend] = {}
        self._lock = threading.RLock()
        self._custom_factories: list[BackendFactory] = []

    @classmethod
    def from_file(cls, config_path: str | Path, model_root: str | Path = ".") -> ModelRegistry:
        """Create from a TOML config file."""
        config = MultimindConfig.from_file(config_path)
        return cls(config, model_root)

    def register_backend_factory(self, factory: BackendFactory) -> None:
        """Register a custom backend factory.

        Factories are tried in registration order before the built-in backends.
        If a factory returns a backend, that backend is used.
        If it returns None, the next factory (or built-in) is tried.
        """
        self._custom_factories.append(factory)

    def model_ids(self) -> list[str]:
        """List all registered model IDs (from config, whether loaded or not)."""
        return [m.id for m in self._config.models]

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded (vs. just configured)."""
        with self._lock:
            return model_id in self._models

    def classify(self, model_id: str, input: ModelInput) -> Verdict:
        """Classify input using a specific model. Loads the model lazily if needed.

        Args:
            model_id: The model to use.
            input: The input to classify.

        Returns:
            A Verdict with label, confidence, and per-class scores.

        Raises:
            KeyError: If no model with that ID exists in config.
            ValueError: If the backend doesn't support the input type.
            RuntimeError: If inference fails.
        """
        with self._lock:
            if model_id in self._models:
                return self._models[model_id].classify(input)

        # Load on first use
        self.load_model(model_id)

        with self._lock:
            return self._models[model_id].classify(input)

    def load_model(self, model_id: str) -> None:
        """Explicitly load a model by ID. No-op if already loaded.

        Raises:
            KeyError: If no model with that ID exists in config.
            RuntimeError: If loading fails.
        """
        with self._lock:
            if model_id in self._models:
                return

            model_config = self._config.get_model(model_id)
            if model_config is None:
                raise KeyError(f"no model with id '{model_id}' in config")

            model_path = self._resolve_path(model_config.path)

            # Try custom factories first
            for factory in self._custom_factories:
                backend = factory(model_config, self._model_root)
                if backend is not None:
                    logger.info(
                        "ModelRegistry: custom backend loaded for %s (%s)",
                        model_id, backend.backend_name(),
                    )
                    self._models[model_id] = backend
                    return

            # Built-in backends
            backend = self._create_builtin_backend(model_id, model_config, model_path)
            self._models[model_id] = backend
            logger.info(
                "ModelRegistry: model loaded %s (%s)", model_id, model_config.backend,
            )

    def reload_model(self, model_id: str, new_path: str | Path) -> None:
        """Hot-reload a model from a new path. The model must already be loaded.

        Raises:
            KeyError: If the model is not loaded.
            RuntimeError: If reload fails.
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"model '{model_id}' not loaded, can't reload")
            self._models[model_id].reload(Path(new_path))
            logger.info("ModelRegistry: model hot-reloaded %s from %s", model_id, new_path)

    def unload_model(self, model_id: str) -> bool:
        """Unload a model, freeing its memory. Returns True if it was loaded."""
        with self._lock:
            removed = self._models.pop(model_id, None) is not None
            if removed:
                logger.info("ModelRegistry: model unloaded %s", model_id)
            return removed

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Get the config for a model."""
        return self._config.get_model(model_id)

    def update_config(self, config: MultimindConfig) -> None:
        """Replace the entire config (e.g. on TOML hot-reload).

        Does NOT unload existing models — they stay loaded until explicitly
        unloaded or the registry is dropped.
        """
        with self._lock:
            self._config = config
            logger.info("ModelRegistry: config updated")

    def register_model(self, model_id: str, backend: ModelBackend) -> None:
        """Register an already-constructed backend under the given model ID.

        Useful for programmatic registration without going through TOML config.
        """
        with self._lock:
            logger.info(
                "ModelRegistry: model registered programmatically %s (%s)",
                model_id, backend.backend_name(),
            )
            self._models[model_id] = backend

    # ── Internal helpers ────────────────────────────────────────────────────

    def _resolve_path(self, config_path: str) -> Path:
        p = Path(config_path)
        if p.is_absolute():
            return p
        return self._model_root / p

    def _create_builtin_backend(
        self, model_id: str, model_config: ModelConfig, model_path: Path,
    ) -> ModelBackend:
        if model_config.backend == "onnx-text":
            from .backends.onnx_text import OnnxTextBackend
            labels = self._load_label_map_int(model_config)
            return OnnxTextBackend(
                model_path=model_path,
                labels=labels,
                min_confidence=model_config.min_confidence,
            )
        elif model_config.backend == "onnx-embed":
            from .backends.onnx_embed import OnnxEmbedBackend
            labels = self._load_label_map_int(model_config)
            return OnnxEmbedBackend(
                model_path=model_path,
                labels=labels,
                embedding_dim=model_config.embedding_dim or 384,
                min_confidence=model_config.min_confidence,
            )
        else:
            raise ValueError(
                f"unsupported backend type '{model_config.backend}' for model '{model_id}'. "
                "Register a custom BackendFactory for non-built-in backends."
            )

    def _load_label_map_int(self, model_config: ModelConfig) -> dict[int, str]:
        """Load labels from a JSON file: {"0": "label_a", "1": "label_b", ...}"""
        if model_config.labels is None:
            return {}
        full_path = self._resolve_path(model_config.labels)
        with open(full_path) as f:
            raw: dict[str, str] = json.load(f)
        return {int(k): v for k, v in raw.items() if k.isdigit() or (k.startswith("-") and k[1:].isdigit())}
