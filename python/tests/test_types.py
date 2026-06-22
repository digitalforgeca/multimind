"""Tests for core types — ModelInput, Verdict, TrainingSignal."""

from multimind.types import ModelInput, InputKind, Verdict, TrainingSignal


def test_model_input_text():
    inp = ModelInput.from_text("hello world")
    assert inp.kind == InputKind.TEXT
    assert inp.text == "hello world"
    assert inp.embedding is None


def test_model_input_embedding():
    inp = ModelInput.from_embedding([0.1, 0.2, 0.3])
    assert inp.kind == InputKind.EMBEDDING
    assert inp.embedding == [0.1, 0.2, 0.3]
    assert inp.text is None


def test_model_input_structured():
    inp = ModelInput.from_structured({"key": "value"})
    assert inp.kind == InputKind.STRUCTURED
    assert inp.structured == {"key": "value"}


def test_verdict():
    v = Verdict(label="safe", confidence=0.95, all_scores={"safe": 0.95, "unsafe": 0.05})
    assert v.label == "safe"
    assert v.confidence == 0.95
    assert len(v.all_scores) == 2


def test_training_signal_round_trip():
    signal = TrainingSignal(
        model_id="test",
        input_text="hello",
        predicted_label="safe",
        corrected_label="unsafe",
        original_confidence=0.72,
    )

    d = signal.to_dict()
    restored = TrainingSignal.from_dict(d)
    assert restored.model_id == "test"
    assert restored.predicted_label == "safe"
    assert restored.corrected_label == "unsafe"
    assert restored.original_confidence == 0.72


def test_training_signal_optional_confidence():
    signal = TrainingSignal(
        model_id="test",
        input_text="hi",
        predicted_label="a",
        corrected_label="b",
    )
    assert signal.original_confidence is None
    d = signal.to_dict()
    assert d["original_confidence"] is None
