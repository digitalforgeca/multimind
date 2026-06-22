# multimind

**Multi-Model Mind** — a generic ONNX model registry with inference, correction signals, and a retrain pipeline for Rust applications.

Multimind has **zero knowledge** of any particular product, domain, or storage layer. Wire it into your own routing, storage, and deployment systems.

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

## Features

| Feature    | Description                                    | Default |
| ---------- | ---------------------------------------------- | ------- |
| `sqlite`   | SQLite signal store via `rusqlite`              | ✅       |
| `postgres` | PostgreSQL signal store via `sqlx`              |         |
| `retrain`  | Background retrain pipeline with artifact export |        |
| `full`     | All of the above                               |         |

## Quick Start

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

## Custom Backends

Implement `ModelBackend` for any inference engine:

```rust
use multimind::{ModelBackend, ModelInput, Verdict};

struct MyApiBackend { /* ... */ }

impl ModelBackend for MyApiBackend {
    fn classify(&self, input: &ModelInput) -> anyhow::Result<Verdict> {
        // Call your API, run your model, etc.
        todo!()
    }
    fn reload(&self, _path: &std::path::Path) -> anyhow::Result<()> { Ok(()) }
    fn backend_name(&self) -> &'static str { "my-api" }
}
```

Register it programmatically:

```rust
registry.register_model("my_model", Box::new(MyApiBackend { /* ... */ }));
```

Or via a `BackendFactory` for config-driven loading:

```rust
registry.register_backend_factory(Box::new(|config, model_root| {
    if config.backend == "my-api" {
        Ok(Some(Box::new(MyApiBackend { /* ... */ })))
    } else {
        Ok(None)
    }
}));
```

## Retrain Pipeline

Enable the `retrain` feature and define your domain-specific weight model:

```rust
use multimind::retrain::{RetrainPipeline, RetrainConfig, WeightModel};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct MyWeights {
    version: u64,
    adjustments: std::collections::HashMap<String, f64>,
}

impl WeightModel for MyWeights {
    fn version(&self) -> u64 { self.version }
    fn set_version(&mut self, v: u64) { self.version = v; }
    fn categories(&self) -> Vec<String> { self.adjustments.keys().cloned().collect() }
    fn adjustment(&self, cat: &str) -> f64 {
        self.adjustments.get(cat).copied().unwrap_or(1.0)
    }
    fn set_adjustment(&mut self, cat: &str, val: f64) {
        self.adjustments.insert(cat.to_string(), val);
    }
}

// Create pipeline with baseline model
let pipeline = RetrainPipeline::new(
    RetrainConfig::default(),
    "my_classifier",
    MyWeights { version: 0, adjustments: Default::default() },
);

// Run manually or start background loop
pipeline.run_retrain(&signal_store, Some(&registry));
pipeline.start_background(signal_store.into(), Some(registry.into()));
```

## Signal Collection

```rust
use multimind::{TrainingSignal, SignalStore};
use multimind::signals::sqlite::SqliteSignalStore;

let store = SqliteSignalStore::open("signals.db").unwrap();

store.record(&TrainingSignal {
    model_id: "classifier".into(),
    input_text: "some input".into(),
    predicted_label: "safe".into(),
    corrected_label: "unsafe".into(),
    original_confidence: Some(0.72),
}).unwrap();

assert_eq!(store.count_pending("classifier").unwrap(), 1);
```

## License

MIT — [Digital Forge Studios](https://dforge.ca)
