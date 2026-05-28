"""
.. _l-plot-onnxruntime-inference:

Running inference with onnxruntime
===================================

This example shows how to build a small ONNX model and run it with
:mod:`onnxruntime`.  The model is a two-layer MLP created with the
:func:`~yaourt.doc.demo_mlp_model` helper.
"""

import numpy as np
import onnxruntime
from yaourt.doc import demo_mlp_model

# %%
# 1. Build the model
# ------------------
#
# :func:`demo_mlp_model <yaourt.doc.demo_mlp_model>` returns an
# ``onnx.ModelProto`` representing a small fully-connected network:
# ``x (3×10) → MatMul → Add → Relu → MatMul → Add → output (3×1)``.
# The input shape is fixed to a batch size of 3.

model = demo_mlp_model("")
print("Opset:", model.opset_import[0].version)
print("Inputs :", [i.name for i in model.graph.input])
print("Outputs:", [o.name for o in model.graph.output])

# %%
# 2. Run inference
# ----------------
#
# We create a random input tensor matching the model's fixed batch size
# and call :meth:`~onnxruntime.InferenceSession.run`.

x = np.random.randn(3, 10).astype(np.float32)

sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
(output,) = sess.run(None, {"x": x})

print("Input  shape:", x.shape)
print("Output shape:", output.shape)
assert output.shape == (3, 1), f"Unexpected output shape: {output.shape}"

# %%
# 3. Run multiple times
# ---------------------
#
# The same session can be reused for multiple calls with different data.

for i in range(3):
    xi = np.random.randn(3, 10).astype(np.float32)
    (yi,) = sess.run(None, {"x": xi})
    print(f"  run={i}  output shape: {yi.shape}")
    assert yi.shape == (3, 1)

print("All runs passed.")
