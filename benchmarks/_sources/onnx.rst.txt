===============
ONNX benchmarks
===============

The following benchmarks compare runtime or *backeend*
with :epkg:`ONNX`.

One-Off predictions
===================

The following benchmark measures the prediction time between
:epkg:`scikit-learn` and :epkg:`onnxruntime` for different configurations
related to *one-off* predictions: predictions are computed
for one observation at a time which is the standard
scenario in a webservice.
:epkg:`onnxruntime` allows for some models to run batch
predictions. If this functionality is available, it is
usually tested for small batches (like 10 observations).

.. toctree::
    :maxdepth: 1

    onnx/onnxruntime_lr
    onnx/onnxruntime_dt
    onnx/onnxruntime_rf
