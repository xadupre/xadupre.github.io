Custom ops
==========

``onnx_light_ort/custom_ops/`` builds ``onnx_light_ort::custom_ops``, a shared
library exporting only ``RegisterCustomOps``. Its current profile registers
``onnx_light::LeakyRelu`` at custom opset 1, with one float32 CPU input and
output and an explicit float ``alpha`` attribute.

This component owns descriptors, domain registration and callback wiring. It
depends on :doc:`bridge_ops_light_to_ort`, which uses :doc:`common_light_to_ort`
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

Python example
--------------

From the repository root on Linux:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   python examples/custom_ops/custom_ops.py \
     --library build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so \
     --model-output build/custom_ops_example.onnx

The example constructs and checks the custom-domain model, loads the plugin,
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
