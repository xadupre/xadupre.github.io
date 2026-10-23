onnx-light called by onnxruntime through a custom operator
==========================================================

``onnx_light_ort::bridge_ops_light_called_by_ort`` is a static library implementing
the custom-op kernel lifecycle. ``CreateLeakyRelu`` reads the required float
alpha and constructs retained state; ``ComputeLeakyRelu`` adapts the current
invocation; ``DestroyLeakyRelu`` releases the state.
``CreateLog``, ``ComputeLog`` and ``DestroyLog`` provide the same lifecycle for
the optimized onnx-light-cpu Log kernel, without attributes.

:doc:`custom_ops` owns registration, descriptors and domain lifetimes.
This bridge owns the ops-specific callbacks, while :doc:`common_light_to_ort`
owns native execution, tensor validation and shared status conversion.
The bridge never depends on the EP bridge or incoming compatibility code.
All C ABI callbacks prevent native exceptions from escaping.

Tests
-----

.. code-block:: bash

   ctest --test-dir build -L bridge_ops_light_called_by_ort --output-on-failure

The native suite in ``tests/test_custom_ops_light_called_by_ort.cc`` verifies explicit zero alpha,
independent retained instances, exception translation and reuse after failure.
It uses the explicit minimal test host, not an installed ORT runtime.
The real-ORT integration suite also carries this label and checks the complete
entry-point/bridge/common chain. See :doc:`custom_ops` for runnable examples.
