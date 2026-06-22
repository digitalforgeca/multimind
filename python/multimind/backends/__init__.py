"""Model backend implementations.

Built-in backends:
- ``onnx_text`` — string-input ONNX models (TF-IDF + SGD pipelines, sklearn exports)
- ``onnx_embed`` — embedding-input ONNX models (pre-computed float32 vectors)

Custom backends can implement the ``ModelBackend`` protocol directly.
"""
