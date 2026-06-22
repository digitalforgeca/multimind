# multimind

**Multi-Model Mind** — a generic ONNX model registry with inference, correction signals, and a retrain pipeline for Python applications.

Multimind has **zero knowledge** of any particular product, domain, or storage layer. Wire it into your own routing, storage, and deployment systems.

Python port of the [Rust multimind crate](https://github.com/digitalforgeca/multimind).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 ModelRegistry                    │
│  ┌────────────┐  ┌────────────┐                 │
│  │ OnnxText   │  │ OnnxEmbed  │  ...custom...   │
│  │ (TF-IDF)   │  │ (384-dim)  │                 │
│  └─────┬──────┘  └─────┬──────┘                 │
│        │ ModelBackend  │                         │
│        └───────┬───────┘                         │
│                ▼                                 │
│         classify(input) → Verdict                │
└────────────────┬────────────────────────────────┘
                 │ correction signals
                 ▼
┌─────────────────────────────────────────────────┐
│              SignalStore                         │
│  ┌────────────┐  ┌────────────┐                 │
│  │  Postgres   │  │   SQLite   │  ...custom...  │
│  └─────────────┘  └────────────┘                │
└────────────────┬────────────────────────────────┘
                 │ batch export
                 ▼
┌─────────────────────────────────────────────────┐
│            RetrainPipeline (optional)            │
│  signals → features → learn → export → hot-swap │
└─────────────────────────────────────────────────┘
```

## Installation

```bash
pip install multimind
```

With PostgreSQL support:

```bash
pip install multimind[postgres]
```

With all extras:

```bash
pip install multimind[full]
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
    def classify(self, input: ModelInput) -> Verdict:
        # Call your API, run your model, etc.
        ...

    def reload(self, path: Path) -> None:
        pass

    def backend_name(self) -> str:
        return "my-api"
```

Register it programmatically:

```python
registry.register_model("my_model", MyApiBackend())
```

Or via a `BackendFactory` for config-driven loading:

```python
def my_factory(config, model_root):
    if config.backend == "my-api":
        return MyApiBackend()
    return None

registry.register_backend_factory(my_factory)
```

## Retrain Pipeline

Define your domain-specific weight model and run the pipeline:

```python
import copy
from multimind.retrain import RetrainPipeline, RetrainConfig, WeightModel

class MyWeights:
    def __init__(self):
        self._version = 0
        self._adjustments: dict[str, float] = {}

    def version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = v

    def categories(self) -> list[str]:
        return list(self._adjustments.keys())

    def adjustment(self, cat: str) -> float:
        return self._adjustments.get(cat, 1.0)

    def set_adjustment(self, cat: str, val: float) -> None:
        self._adjustments[cat] = val

    def __deepcopy__(self, memo):
        new = MyWeights()
        new._version = self._version
        new._adjustments = dict(self._adjustments)
        return new

# Create pipeline with baseline model
pipeline = RetrainPipeline(
    RetrainConfig(),
    "my_classifier",
    MyWeights(),
)

# Run manually or start background loop
pipeline.run_retrain(signal_store)
pipeline.start_background(signal_store, registry)
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

assert store.count_pending("classifier") == 1
```

## Requirements

- Python 3.11+
- `numpy` and `onnxruntime` for ONNX inference
- `psycopg2` (optional) for PostgreSQL signal store

## License

MIT — [Digital Forge Studios](https://dforge.ca)
