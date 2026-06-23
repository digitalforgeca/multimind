"""Tests for SQLite signal store — mirrors Rust sqlite.rs tests."""

from multimind.types import TrainingSignal
from multimind.signals.sqlite import SqliteSignalStore


def test_round_trip():
    store = SqliteSignalStore.in_memory()

    signal = TrainingSignal(
        model_id="test_model",
        input_text="test content",
        predicted_label="reject",
        corrected_label="store",
        original_confidence=0.52,
    )

    store.record(signal)
    assert store.count_pending("test_model") == 1
    assert store.count_pending("other_model") == 0

    signals = store.export_pending("test_model")
    assert len(signals) == 1
    assert signals[0].predicted_label == "reject"
    assert signals[0].corrected_label == "store"
    assert signals[0].signal_id is not None, "exported signals must have IDs"

    # Targeted consume: only mark the exported batch
    ids = [s.signal_id for s in signals if s.signal_id is not None]
    store.mark_consumed("test_model", ids)
    assert store.count_pending("test_model") == 0


def test_targeted_consume_doesnt_eat_new_signals():
    store = SqliteSignalStore.in_memory()

    # Record 2 signals, export them
    for i in range(2):
        store.record(TrainingSignal(
            model_id="m",
            input_text=f"old {i}",
            predicted_label="a",
            corrected_label="b",
        ))
    batch = store.export_pending("m")
    ids = [s.signal_id for s in batch if s.signal_id is not None]
    assert len(ids) == 2

    # A new signal arrives between export and consume
    store.record(TrainingSignal(
        model_id="m",
        input_text="new arrival",
        predicted_label="x",
        corrected_label="y",
    ))
    assert store.count_pending("m") == 3

    # Targeted consume only marks the 2 exported signals
    store.mark_consumed("m", ids)
    assert store.count_pending("m") == 1

    # The surviving signal is the new one
    remaining = store.export_pending("m")
    assert remaining[0].input_text == "new arrival"


def test_mark_all_consumed_drains_everything():
    store = SqliteSignalStore.in_memory()

    for i in range(5):
        store.record(TrainingSignal(
            model_id="m",
            input_text=f"input {i}",
            predicted_label="a",
            corrected_label="b",
        ))
    assert store.count_pending("m") == 5

    store.mark_all_consumed("m")
    assert store.count_pending("m") == 0


def test_multiple_models_isolated():
    store = SqliteSignalStore.in_memory()

    for i in range(3):
        store.record(TrainingSignal(
            model_id="model_a",
            input_text=f"input {i}",
            predicted_label="class_1",
            corrected_label="class_2",
            original_confidence=0.6,
        ))

    store.record(TrainingSignal(
        model_id="model_b",
        input_text="another input",
        predicted_label="x",
        corrected_label="y",
        original_confidence=None,
    ))

    assert store.count_pending("model_a") == 3
    assert store.count_pending("model_b") == 1

    # Targeted consume of model_a doesn't affect model_b
    batch = store.export_pending("model_a")
    ids = [s.signal_id for s in batch if s.signal_id is not None]
    store.mark_consumed("model_a", ids)
    assert store.count_pending("model_a") == 0
    assert store.count_pending("model_b") == 1


def test_export_with_limit():
    store = SqliteSignalStore.in_memory()

    for i in range(10):
        store.record(TrainingSignal(
            model_id="test",
            input_text=f"input {i}",
            predicted_label="a",
            corrected_label="b",
            original_confidence=0.5,
        ))

    # Without limit: get all 10
    all_signals = store.export_pending("test")
    assert len(all_signals) == 10
    # All exported signals have IDs
    assert all(s.signal_id is not None for s in all_signals)

    # With limit: get exactly 3
    limited = store.export_pending("test", limit=3)
    assert len(limited) == 3

    # Limit larger than available: get all
    big_limit = store.export_pending("test", limit=100)
    assert len(big_limit) == 10
