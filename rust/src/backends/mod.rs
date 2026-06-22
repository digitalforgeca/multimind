//! Model backend implementations.
//!
//! Built-in backends:
//! - [`onnx_text`] — string-input ONNX models (TF-IDF + SGD pipelines, sklearn exports)
//! - [`onnx_embed`] — embedding-input ONNX models (pre-computed float32 vectors)
//!
//! Custom backends can implement [`ModelBackend`](crate::ModelBackend) directly.

pub mod onnx_text;
pub mod onnx_embed;
