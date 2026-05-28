yet-another-onnxruntime-extensions
===================================

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_core.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_core.yml
    :alt: core

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_ortops.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_ortops.yml
    :alt: ortops

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/build.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/build.yml
    :alt: build

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/mypy.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/mypy.yml
    :alt: mypy

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/docs.yml
    :alt: Documentation

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/style.yml
    :alt: Style

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/pyrefly.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/pyrefly.yml
    :alt: pyrefly

.. image:: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/spelling.yml
    :alt: Spelling

.. image:: https://codecov.io/gh/xadupre/yet-another-onnxruntime-extensions/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/yet-another-onnxruntime-extensions

.. image:: https://img.shields.io/github/repo-size/xadupre/yet-another-onnxruntime-extensions
    :target: https://github.com/xadupre/yet-another-onnxruntime-extensions

**yet-another-onnxruntime-extensions** (``yaourt``) is an experimental library
of `ONNX Runtime <https://onnxruntime.ai/>`_ extensions: custom C++ operators,
profiling utilities, and plotting helpers.
The source code is available on
`GitHub <https://github.com/xadupre/yet-another-onnxruntime-extensions>`_.

Installation
------------

.. code-block:: bash

    pip install yet-another-onnxruntime-extensions

.. note::

    The pre-built wheel includes sparse CPU operators only.
    Fused-kernel CUDA operators must be compiled from source with a
    CUDA-enabled CMake build — see :doc:`getting_started` for instructions.

.. code-block:: python

    import yaourt
    print(yaourt.__version__)

Key Features
------------

:mod:`yaourt.ortops` — Custom C++ Operators
    Sparse CPU operators are shipped as pre-built binaries inside the wheel
    and registered with ONNX Runtime via
    :data:`~yaourt.ortops.SPARSE_CPU_LIB_PATH`.
    Fused-kernel CUDA operators (:data:`~yaourt.ortops.FUSED_KERNEL_CUDA_LIB_PATH`)
    require a CUDA-enabled build — see :doc:`getting_started` for CMake
    build instructions.

:mod:`yaourt.tools` — Profiling Tools
    Parse ONNX Runtime JSON profiling output into ``pandas`` DataFrames
    (:func:`~yaourt.tools.js_profile.js_profile_to_dataframe`) and visualize
    per-operator timings and execution timelines with matplotlib
    (:func:`~yaourt.tools.js_profile.plot_ort_profile`,
    :func:`~yaourt.tools.js_profile.plot_ort_profile_timeline`).

:mod:`yaourt.plot` — Benchmark and Plot Helpers
    Horizontal benchmark comparison histograms with error bars
    (:func:`~yaourt.plot.benchmark.hhistograms`) and tensor histogram
    utilities for model analysis (:func:`~yaourt.doc.plot_histogram`).

:mod:`yaourt.reference` — Reference Evaluator
    A pure-Python ONNX evaluator useful for testing and debugging custom
    operators without a full ONNX Runtime build.

Quick Start
-----------

Run inference with ONNX Runtime:

.. code-block:: python

    import numpy as np
    import onnxruntime
    from yaourt.doc import demo_mlp_model

    model = demo_mlp_model("")  # filename argument is unused
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.random.randn(3, 10).astype(np.float32)
    (output,) = sess.run(None, {"x": x})
    print("Output shape:", output.shape)

Load the custom C++ operators:

.. code-block:: python

    import onnxruntime as ort
    from yaourt.ortops import SPARSE_CPU_LIB_PATH

    opts = ort.SessionOptions()
    opts.register_custom_ops_library(str(SPARSE_CPU_LIB_PATH))

.. toctree::
    :maxdepth: 1
    :caption: Contents

    getting_started
    custom_ops/index
    api/index
    auto_examples/index

.. only:: not ci_build

    .. toctree::
        :maxdepth: 1
        :caption: Miscellaneous

        ci_durations
