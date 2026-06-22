"""Tests for TOML config parsing — mirrors Rust config.rs tests."""

from multimind.config import MultimindConfig


def test_parse_minimal_config():
    toml = '''
        [[models]]
        id = "classifier"
        backend = "onnx-text"
        path = "models/classifier.onnx"
    '''
    config = MultimindConfig.from_toml(toml)
    assert len(config.models) == 1
    assert config.models[0].id == "classifier"
    assert config.models[0].backend == "onnx-text"
    assert config.models[0].min_confidence == 0.5


def test_parse_full_config():
    toml = '''
        [[models]]
        id = "vibeguard"
        backend = "onnx-text"
        path = "models/vibeguard.onnx"
        labels = "models/vibeguard_labels.json"
        classes = ["SAFE", "UNSAFE", "REVIEW"]
        min_confidence = 0.7

        [models.retrain]
        min_signals = 50
        min_sessions = 100
    '''
    config = MultimindConfig.from_toml(toml)
    m = config.models[0]
    assert m.id == "vibeguard"
    assert m.classes is not None and len(m.classes) == 3
    assert m.min_confidence == 0.7
    assert m.retrain is not None
    assert m.retrain.min_signals == 50


def test_parse_multi_model_config():
    toml = '''
        [[models]]
        id = "sivu"
        backend = "onnx-text"
        path = "models/sivu.onnx"

        [[models]]
        id = "sicu"
        backend = "onnx-embed"
        path = "models/sicu.onnx"
        embedding_dim = 384
    '''
    config = MultimindConfig.from_toml(toml)
    assert len(config.models) == 2
    sivu = config.get_model("sivu")
    assert sivu is not None and sivu.backend == "onnx-text"
    sicu = config.get_model("sicu")
    assert sicu is not None and sicu.embedding_dim == 384
    assert config.get_model("nonexistent") is None
