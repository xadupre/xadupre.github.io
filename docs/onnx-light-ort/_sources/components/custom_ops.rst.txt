Custom ops
==========

``onnx_light_ort/custom_ops/`` builds ``onnx_light_ort::custom_ops``, a shared
library exporting only ``RegisterCustomOps``. Its current profile registers
``onnx_light::LeakyRelu`` at custom opset 1, with one floating-point CPU input and
output and an explicit float ``alpha`` attribute.
It also registers ``onnx_light_cpu::Log`` at custom opset 1, with the same
CPU input/output profile and no attributes. The numerical implementations
are onnx-light-cpu's ``LogFloat32``, ``LogFloat64``, ``LogFloat16`` and
``LogBFloat16WithTuning``, using the dependency's default tuning and dispatch.
Both operators support changing shapes, scalars and empty tensors up to rank 16.

Each operator exposes ``float16``, ``bfloat16``, ``float32`` and ``float64``.
Four concrete typed descriptors share the same domain, name and custom opset;
ORT selects the matching implementation from the model's element types.
Input and output types must match. There is no bridge-level conversion or
fallback to another operator. LeakyRelu delegates all four types to the native
onnx-light kernel; its ``alpha`` attribute remains an ONNX float.

This component owns descriptors, domain registration and callback wiring. It
depends on :doc:`bridge_ops_light_called_by_ort`, which uses :doc:`common_light_to_ort`
for native state, tensor validation and status conversion, never
on the opposite bridge or the EP library. It requests the supplied API 18
table and has no link dependency on ORT. See :doc:`../architecture` for domain,
kernel and borrowed-buffer lifetimes.

Tests
-----

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L custom_ops --output-on-failure

``tests/test_custom_ops.cc`` calls the real shared-library export with an
explicit minimal C API test host. It verifies stable domain reuse, failed
registration, unsupported API negotiation, failed construction clearing its
result, native execution through callbacks, exception-to-status translation,
metadata cleanup and reuse after failure. No ORT runtime or mocking framework
is needed. This fixture is not the production compatibility bridge.

With ``ONNX_LIGHT_ORT_TEST_WITH_ORT=ON``, the same label also selects the
real-ORT integration suite and runnable examples.
The real-ORT and standalone suites exercise all four types for both operators,
including changing shapes, scalars, empty tensors, non-finite values and
float64 values that cannot be represented as float32.
The ORT 1.18 Python API cannot construct BFLOAT16 OrtValues directly. Its tests
use float32 graph inputs/outputs with explicit Cast nodes around the BFLOAT16
custom node, whose intermediate types are explicitly declared. The native
custom kernel still receives and produces BFLOAT16, without changing either
bridge or skipping this type. The standalone suite uses BFLOAT16 arrays directly.

BFLOAT16 with ORT's Python API
------------------------------

To avoid NumPy feed/output dtype-mapping limitations for ``ml_dtypes.bfloat16``,
bind typed OrtValues over the raw 16-bit storage. The session's model
input/output types must be ``TensorProto.BFLOAT16``:

This snippet requires an ORT Python build providing
``ortvalue_from_numpy_with_onnx_type``; that convenience method is absent in
ORT 1.18.1. With that older Python interface, use explicit Cast nodes at the
graph boundaries as in the integration tests, or construct BFLOAT16 tensors
through the C API's ``CreateTensorWithDataAsOrtValue``.

.. code-block:: python

   import numpy
   import onnxruntime
   from ml_dtypes import bfloat16
   from onnx_light.onnx import TensorProto

   values = numpy.array([-4, 2], dtype=bfloat16)
   output = numpy.empty_like(values)
   input_value = onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
       values.view(numpy.uint16), TensorProto.BFLOAT16
   )
   output_value = onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
       output.view(numpy.uint16), TensorProto.BFLOAT16
   )
   binding = session.io_binding()
   binding.bind_ortvalue_input("X", input_value)
   binding.bind_ortvalue_output("Y", output_value)
   session.run_with_iobinding(binding)

``output`` retains the bfloat16 dtype. The standalone ``CustomOpSessionOrtCalledByLight`` accepts
these arrays directly. Keep bound arrays alive until inference completes.

Python example
--------------

From the repository root on Linux:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   export PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"
   python examples/custom_ops/custom_ops.py \
     --library build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so \
     --model-output build/custom_ops_example.onnx

The example constructs and checks the custom-domain model with
``onnx_light.onnx`` (not the ``onnx`` package), loads the plugin,
runs inference and checks the result ``[-0.5, -0.125, 0, 2]``. The saved model
can then be consumed by the C++ example.

.. literalinclude:: ../../examples/custom_ops/custom_ops.py
   :language: python

C++ example
-----------

The native build compiles the example without installing or linking ORT.
At execution, provide the absolute path to an existing ORT shared library,
the custom-op library and the model saved above:

.. code-block:: bash

   build/examples/custom_ops/onnx_light_ort_custom_ops_example \
     /absolute/path/to/libonnxruntime.so \
     build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so \
     build/custom_ops_example.onnx

The example is a real ORT application, so it deliberately loads ORT and resolves
``OrtGetApiBase``. This is not a fallback in either production bridge.
It registers the custom-op library through ``RegisterCustomOpsLibrary_V2``,
creates a real session and verifies the same output. Scoped owners release
values, sessions and options before the runtime library unloads.

Use the platform-specific library suffix and executable location on macOS or
Windows. Multi-configuration generators additionally use ``Release/``.
Set ``ONNX_LIGHT_ORT_BUILD_EXAMPLES=OFF`` to disable the C++ example build.

To run both examples through CTest, enable real-host tests and pass
``-DONNX_LIGHT_ORT_RUNTIME_LIBRARY=/absolute/path/to/the/ORT/library``.
CTest first runs the Python example to generate the model, then the C++
example; fixture dependencies also apply when selecting only the C++ test.

.. literalinclude:: ../../examples/custom_ops/custom_ops.cc
   :language: cpp

CPU Log in both custom-op hosts
-------------------------------

The :doc:`outgoing gallery example <../auto_examples/plot_custom_ops_light_called_by_ort>`
executes ``onnx_light_cpu::Log`` in real ORT. The
:doc:`incoming gallery example <../auto_examples/plot_custom_ops_ort_called_by_light>`
loads the same custom-op library through the standalone onnx-light host,
without importing or loading ORT. Neither path uses an EP.
Only onnx-light-cpu's standalone ``ml_cpu_compute`` subproject is configured,
not its Python bindings or another onnx-light runtime.

Reproduce the comparison from the repository root after building both bridges
and the onnx-light model bindings in Release mode:

.. code-block:: bash

   export PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"
   python docs/examples/plot_benchmark_log_custom_ops.py \
     --library build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so \
     --host build/onnx_light_ort/bridge_ops_ort_called_by_light/libonnx_light_ort_bridge_ops_ort_called_by_light.so \
     --output build/cpu-custom-ops-log-benchmark.json

The script compares steady-state ORT I/O binding calls with preallocated buffers
and standalone onnx-light ``run`` calls with protective input/output copies.
It measures Python interfaces, not isolated arithmetic, over eleven sizes
from 1 to 1,048,576 elements. NumPy is not a timed backend.
Both custom-op directions are labelled explicitly; speedup is the median
ORT built-in Log latency divided by the path's median latency at the same
input size and node count. Separate latency and speedup plots show one or ten
Log nodes; the standalone host supports only one node. It uses identical
float32 inputs from ``default_rng(42).uniform(0.01, 10)`` and validates complete
outputs against NumPy float64 Log rounded to float32, before and after timing.
Each backend runs in its own child process, with sequential, one-thread ORT
sessions and no private CPU-kernel worker pool. Three trials rotate backend
order; each size receives ten warmups followed by 31 timed batches of 20 calls.
The report retains every per-call batch sample and records versions, affinity
and measurement settings. The table reports the median of the 93 batch samples.

The following historical diagnostic results used ``session.run``, including
output allocation, rather than the current I/O binding setup. They used a
Xeon Platinum 8480C, affinity
restricted to CPU 0 with ``taskset -c 0``, GCC 13.2.0, Release mode, runtime SIMD
dispatch, Python 3.12.15 and source-built ORT 1.32.0. The onnx-light revision was
``5c899bdd2c1011e2825728b4fc950dcc88fc6aeb`` and onnx-light-cpu was
``054cdd984d9f85a7692dc4f678b2f13275e8ede1``.

.. list-table:: Steady-state Log inference, microseconds per call
   :header-rows: 1

   * - Elements
     - Standard ORT Log
     - CPU custom op called by ORT
     - ORT / custom op
     - CPU custom op called by standalone onnx-light
   * - 1,024
     - 5.076
     - 4.700
     - 1.08x
     - 15.596
   * - 32,768
     - 52.071
     - 16.833
     - 3.09x
     - 42.551
   * - 65,536
     - 100.364
     - 30.519
     - 3.29x
     - 81.738
   * - 262,144
     - 395.220
     - 118.298
     - 3.34x
     - 437.091
   * - 1,048,576
     - 1565.080
     - 458.078
     - 3.42x
     - 1919.574

For 1,048,576 elements, the three trial medians ranged from 1558.594 to
1575.189 microseconds for standard ORT and from 444.595 to 458.625 microseconds
for the custom op called by ORT. These machine-specific measurements are not
portable thresholds or a multithreaded performance claim. Small tensors can
be dominated by dispatch overhead. The standalone Python interface includes
protective input/output copies and is slower than ORT for the larger sizes;
its timings must not be presented as the outgoing bridge's speedup.
