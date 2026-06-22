# multimind

**Multi-Model Mind** — a generic ONNX model registry with inference, correction signals, and a retrain pipeline.

Dual implementation: **Rust** (`rust/`) and **Python** (`python/`). Both expose the same architecture, traits/protocols, and signal stores. Multimind has **zero knowledge** of any particular product, domain, or storage layer — wire it into your own routing, storage, and deployment systems.

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
                 │ batch export (with signal IDs)
                 ▼
┌─────────────────────────────────────────────────┐
│            RetrainPipeline (optional)            │
│  signals → features → learn → export → hot-swap │
└─────────────────────────────────────────────────┘
```

## Repo Layout

Each subdirectory is independently installable with its own README, LICENSE, and build config.

```
multimind/
├── rust/                 # Rust crate (cargo build / crates.io)
│   ├── Cargo.toml
│   ├── README.md
│   ├── LICENSE
│   └── src/
│       ├── lib.rs        # Core types, traits (SignalStore, ModelBackend)
│       ├── config.rs     # TOML config parsing
│       ├── registry.rs   # ModelRegistry
│       ├── backends/     # ONNX inference backends
│       ├── signals/      # SQLite + Postgres signal stores
│       └── retrain/      # Pipeline, weight learning, artifacts
├── python/               # Python package (pip install / PyPI)
│   ├── pyproject.toml
│   ├── README.md
│   ├── LICENSE
│   ├── multimind/        # Package source (mirrors Rust module structure)
│   └── tests/            # pytest suite
├── LICENSE
└── README.md
```

### Install just one language

**Python only:**
```bash
git clone https://github.com/digitalforgeca/multimind.git
cd multimind/python
pip install -e ".[dev]"
pytest -v
```

**Rust only:**
```bash
git clone https://github.com/digitalforgeca/multimind.git
cd multimind/rust
cargo build --features full
cargo test --features full
```

Each subdirectory is a complete, self-contained project — no cross-directory dependencies.

## Core Concepts

**ModelBackend** — any inference engine that takes an input and returns a `Verdict` (label + confidence + per-class scores). Built-in: ONNX text (TF-IDF) and ONNX embedding (384-dim). Implement the trait/protocol for custom backends.

**SignalStore** — records correction signals (`TrainingSignal`) and exports them for retraining. Built-in: SQLite and PostgreSQL. Signal consumption is **ID-targeted** — `mark_consumed(model_id, signal_ids)` only marks the specific rows from the exported batch, preventing race conditions with newly-arrived signals. `mark_all_consumed` is available for explicit drain operations.

**RetrainPipeline** — optional background loop that watches signal accumulation, runs retrain cycles (feature extraction → weight learning → artifact export), and hot-swaps models in the registry.

---

## Rust

### Features

| Feature    | Description                                      | Default |
| ---------- | ------------------------------------------------ | ------- |
| `sqlite`   | SQLite signal store via `rusqlite`               | ✅       |
| `postgres` | PostgreSQL signal store via `sqlx`               |         |
| `retrain`  | Background retrain pipeline with artifact export |         |
| `full`     | All of the above                                 |         |

### Quick Start

```toml
[dependencies]
multimind = "0.1"
```

```rust
use multimind::{ModelRegistry, MultimindConfig, ModelInput};

let config = MultimindConfig::from_toml(r#"
    [[models]]
    id = "classifier"
    backend = "onnx-text"
    path = "models/classifier.onnx"
    labels = "models/labels.json"
"#).unwrap();

let registry = ModelRegistry::new(config, ".");
let verdict = registry.classify("classifier", &ModelInput::Text("hello world".into())).unwrap();
println!("{}: {:.2}", verdict.label, verdict.confidence);
```

### Custom Backends

```rust
use multimind::{ModelBackend, ModelInput, Verdict};

struct MyApiBackend { /* ... */ }

impl ModelBackend for MyApiBackend {
    fn classify(&self, input: &ModelInput) -> anyhow::Result<Verdict> { todo!() }
    fn reload(&self, _path: &std::path::Path) -> anyhow::Result<()> { Ok(()) }
    fn backend_name(&self) -> &'static str { "my-api" }
}

registry.register_model("my_model", Box::new(MyApiBackend { /* ... */ }));
```

### Signal Collection

```rust
use multimind::{TrainingSignal, SignalStore};
use multimind::signals::sqlite::SqliteSignalStore;

let store = SqliteSignalStore::open("signals.db").unwrap();

store.record(&TrainingSignal {
    signal_id: None,
    model_id: "classifier".into(),
    input_text: "some input".into(),
    predicted_label: "safe".into(),
    corrected_label: "unsafe".into(),
    original_confidence: Some(0.72),
}).unwrap();

// Export → retrain → targeted consume
let batch = store.export_pending("classifier", Some(100)).unwrap();
let ids: Vec<String> = batch.iter().filter_map(|s| s.signal_id.clone()).collect();
// ... retrain with batch ...
store.mark_consumed("classifier", &ids).unwrap();
```

### Retrain Pipeline

```rust
use multimind::retrain::{RetrainPipeline, RetrainConfig, WeightModel};

let pipeline = RetrainPipeline::new(
    RetrainConfig::default(),
    "my_classifier",
    MyWeights { version: 0, adjustments: Default::default() },
);

// Synchronous or background
pipeline.run_retrain(&signal_store, Some(&registry));
pipeline.start_background(signal_store.into(), Some(registry.into()));
```

---

## Python

### Installation

```bash
pip install multimind              # core + SQLite
pip install multimind[postgres]    # + PostgreSQL
pip install multimind[full]        # all extras
```

### Quick Start

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

### Custom Backends

```python
from pathlib import Path
from multimind import ModelBackend, ModelInput, Verdict

class MyApiBackend:
    def classify(self, input: ModelInput) -> Verdict: ...
    def reload(self, path: Path) -> None: pass
    def backend_name(self) -> str: return "my-api"

registry.register_model("my_model", MyApiBackend())
```

### Signal Collection

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

### Retrain Pipeline

```python
from multimind.retrain import RetrainPipeline, RetrainConfig

pipeline = RetrainPipeline(RetrainConfig(), "my_classifier", MyWeights())

pipeline.run_retrain(signal_store)
pipeline.start_background(signal_store, registry)
```

### Requirements

- Python 3.11+
- `numpy` + `onnxruntime` for ONNX inference
- `psycopg2` (optional) for PostgreSQL

---

## Running Tests

```bash
# Rust
cd rust && cargo test --features full

# Python
cd python && pip install -e ".[dev]" && pytest -v
```

## License

MIT — [Digital Forge Studios](https://dforge.ca)
