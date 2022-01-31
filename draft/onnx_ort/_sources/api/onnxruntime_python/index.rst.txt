
Summary of onnxruntime API
==========================

Most of the code in :epkg:`onnxruntime` is written in C++ and exposed
in Python using :epkg:`pybind11`. For inference, the main class
is :epkg:`InferenceSession`. It wraps C class :ref:`l-ort-inference-session-c`.
The python class is easier to use. Both have the same name.
It adds some short overhead but significant on small models
such as a linear regression.

.. toctree::
    :maxdepth: 1

    helpers
    ortvalue
    sparse
    inference
    exceptions
