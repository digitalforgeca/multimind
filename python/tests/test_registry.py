"""Tests for ModelRegistry — custom backends, config operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimind.config import MultimindConfig
from multimind.types import ModelInput, InputKind, Verdict
from multimind.registry import ModelRegistry


class MockBackend:
    """A simple mock backend for testing registry operations."""

    def __init__(self, label: str = "mock_label", confidence: float = 0.99) -> None:
        self._label = label
        self._confidence = confidence

    def classify(self, input: ModelInput) -> Verdict:
        return Verdict(
            label=self._label,
            confidence=self._confidence,
            all_scores={self._label: self._confidence},
        )

    def reload(self, path: Path) -> None:
        pass

    def backend_name(self) -> str:
        return "mock"


def test_register_model_programmatic():
    config = MultimindConfig()
    registry = ModelRegistry(config)

    registry.register_model("test", MockBackend())
    assert registry.is_loaded("test")

    verdict = registry.classify("test", ModelInput.from_text("hello"))
    assert verdict.label == "mock_label"
    assert verdict.confidence == 0.99


def test_model_ids():
    toml = '''
        [[models]]
        id = "a"
        backend = "onnx-text"
        path = "a.onnx"

        [[models]]
        id = "b"
        backend = "onnx-embed"
        path = "b.onnx"
    '''
    config = MultimindConfig.from_toml(toml)
    registry = ModelRegistry(config)
    ids = registry.model_ids()
    assert "a" in ids
    assert "b" in ids


def test_unknown_model_raises():
    config = MultimindConfig()
    registry = ModelRegistry(config)

    with pytest.raises(KeyError):
        registry.classify("nonexistent", ModelInput.from_text("test"))


def test_unload_model():
    config = MultimindConfig()
    registry = ModelRegistry(config)

    registry.register_model("test", MockBackend())
    assert registry.is_loaded("test")

    removed = registry.unload_model("test")
    assert removed
    assert not registry.is_loaded("test")

    # Unloading again returns False
    assert not registry.unload_model("test")


def test_update_config():
    config1 = MultimindConfig.from_toml('''
        [[models]]
        id = "old"
        backend = "onnx-text"
        path = "old.onnx"
    ''')
    registry = ModelRegistry(config1)
    assert "old" in registry.model_ids()

    config2 = MultimindConfig.from_toml('''
        [[models]]
        id = "new"
        backend = "onnx-text"
        path = "new.onnx"
    ''')
    registry.update_config(config2)
    assert "new" in registry.model_ids()
    assert "old" not in registry.model_ids()


def test_custom_backend_factory():
    toml = '''
        [[models]]
        id = "custom"
        backend = "my-special"
        path = "custom.onnx"
    '''
    config = MultimindConfig.from_toml(toml)
    registry = ModelRegistry(config)

    def factory(model_config, model_root):
        if model_config.backend == "my-special":
            return MockBackend(label="custom_result", confidence=0.88)
        return None

    registry.register_backend_factory(factory)

    verdict = registry.classify("custom", ModelInput.from_text("test"))
    assert verdict.label == "custom_result"
    assert verdict.confidence == 0.88


def test_unsupported_backend_raises():
    toml = '''
        [[models]]
        id = "bad"
        backend = "unknown-backend"
        path = "bad.onnx"
    '''
    config = MultimindConfig.from_toml(toml)
    registry = ModelRegistry(config)

    with pytest.raises(ValueError, match="unsupported backend"):
        registry.load_model("bad")
