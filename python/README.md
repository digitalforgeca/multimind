# multimind

**Multi-Model Mind** — a generic ONNX model registry with inference, correction signals, and a retrain pipeline for Python applications.

Multimind has **zero knowledge** of any particular product, domain, or storage layer. Wire it into your own routing, storage, and deployment systems.

Part of the [multimind monorepo](https://github.com/digitalforgeca/multimind). A parallel Rust implementation lives at [`rust/`](https://github.com/digitalforgeca/multimind/tree/main/rust).

## Installation

```bash
pip install multimind              # core + SQLite
pip install multimind[postgres]    # + PostgreSQL
pip install multimind[full]        # all extras
```

Or install from source:

```bash
git clone https://github.com/digitalforgeca/multimind.git
cd multimind/python
pip install -e ".[dev]"
```

## Quick Start

```python
from multimind import ModelRegistry, MultimindConfig, ModelInput

config = MultimindConfig.from_toml('''
    [[models]]
    id = "classifier"
    backend = "onnx-text"
    path = "models/classifier.onnx"
    labels = "models/labels.json"
''')

registry = ModelRegistry(config, ".")
verdict = registry.classify("classifier", ModelInput.from_text("hello world"))
print(f"{verdict.label}: {verdict.confidence:.2f}")
```

## Custom Backends

Implement the `ModelBackend` protocol for any inference engine:

```python
from pathlib import Path
from multimind import ModelBackend, ModelInput, Verdict

class MyApiBackend:
    def classify(self, input: ModelInput) -> Verdict: ...
    def reload(self, path: Path) -> None: pass
    def backend_name(self) -> str: return "my-api"

registry.register_model("my_model", MyApiBackend())
```

## Signal Collection

```python
from multimind import TrainingSignal
from multimind.signals.sqlite import SqliteSignalStore

store = SqliteSignalStore.open("signals.db")

store.record(TrainingSignal(
    model_id="classifier",
    input_text="some input",
    predicted_label="safe",
    corrected_label="unsafe",
    original_confidence=0.72,
))

# Export → retrain → targeted consume
batch = store.export_pending("classifier", limit=100)
ids = [s.signal_id for s in batch if s.signal_id]
# ... retrain with batch ...
store.mark_consumed("classifier", ids)
```

Signal consumption is **ID-targeted** — `mark_consumed(model_id, signal_ids)` only marks the specific rows from the exported batch, preventing race conditions with newly-arrived signals. `mark_all_consumed` is available for explicit drain operations.

## Retrain Pipeline

```python
from multimind.retrain import RetrainPipeline, RetrainConfig

pipeline = RetrainPipeline(RetrainConfig(), "my_classifier", MyWeights())

pipeline.run_retrain(signal_store)
pipeline.start_background(signal_store, registry)
```

## Running Tests

```bash
pytest -v
```

## Requirements

- Python 3.11+
- `numpy` + `onnxruntime` for ONNX inference
- `psycopg2` (optional) for PostgreSQL

## License

MIT — [Digital Forge Studios](https://dforge.ca)
