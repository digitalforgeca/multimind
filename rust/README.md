# multimind

**Multi-Model Mind** — a generic ONNX model registry with inference, correction signals, and a retrain pipeline for Rust applications.

Multimind has **zero knowledge** of any particular product, domain, or storage layer. Wire it into your own routing, storage, and deployment systems.

Part of the [multimind monorepo](https://github.com/digitalforgeca/multimind). A parallel Python implementation lives at [`python/`](https://github.com/digitalforgeca/multimind/tree/main/python).

## Features

| Feature    | Description                                      | Default |
| ---------- | ------------------------------------------------ | ------- |
| `sqlite`   | SQLite signal store via `rusqlite`               | ✅       |
| `postgres` | PostgreSQL signal store via `sqlx`               |         |
| `retrain`  | Background retrain pipeline with artifact export |         |
| `full`     | All of the above                                 |         |

## Quick Start

```toml
[dependencies]
multimind = "0.1"
```

Or build from source:

```bash
git clone https://github.com/digitalforgeca/multimind.git
cd multimind/rust
cargo build --features full
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

## Custom Backends

Implement `ModelBackend` for any inference engine:

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

## Signal Collection

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

Signal consumption is **ID-targeted** — `mark_consumed(model_id, signal_ids)` only marks the specific rows from the exported batch, preventing race conditions with newly-arrived signals. `mark_all_consumed` is available for explicit drain operations.

## Retrain Pipeline

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

## Running Tests

```bash
cargo test --features full
```

## License

MIT — [Digital Forge Studios](https://dforge.ca)
