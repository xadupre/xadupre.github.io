onnx-light called by onnxruntime through a custom operator
==========================================================

Run native onnx-light ``LeakyRelu`` and onnx-light-cpu ``Log`` kernels inside
ONNX Runtime through custom-domain operators. This project is implemented.

Design
------

``onnx_light_ort::bridge_ops_light_called_by_ort`` is a static library implementing
the custom-op kernel lifecycle. ``CreateLeakyRelu`` reads the required float
alpha and constructs retained state; ``ComputeLeakyRelu`` adapts the current
invocation; ``DestroyLeakyRelu`` releases the state.
``CreateLog``, ``ComputeLog`` and ``DestroyLog`` provide the same lifecycle for
the optimized onnx-light-cpu Log kernel, without attributes.

``custom_ops`` owns registration, descriptors and domain lifetimes.
This bridge owns the ops-specific callbacks, while ``common_light_to_ort``
owns native execution, tensor validation and shared status conversion.
The bridge never depends on the EP bridge or incoming compatibility code.
All C ABI callbacks prevent native exceptions from escaping.

Requirements
------------

Build with a C++20 compiler and CMake 3.28 or newer, with
``ONNX_LIGHT_ORT_BUILD_PLUGIN=ON``. CMake supplies onnx-light, the standalone
onnx-light-cpu kernels and pinned ORT headers. The plugin does not link ORT,
but execution requires a real ORT host supporting API 18.
Python examples additionally need ``requirements/ort-tests.txt`` and the
matching ``onnx_light.onnx`` bindings from ``build/python``.
See :doc:`../development` for source-build instructions.

Tests
-----

.. code-block:: bash

   ctest --test-dir build -L bridge_ops_light_called_by_ort --output-on-failure

The native suite in ``tests/test_custom_ops_light_called_by_ort.cc`` verifies explicit zero alpha,
independent retained instances, exception translation and reuse after failure.
It uses the explicit minimal test host, not an installed ORT runtime.
The real-ORT integration suite also carries this label and checks the complete
entry-point/bridge/common chain.

.. include:: common_light_to_ort.inc

.. include:: custom_ops.inc
