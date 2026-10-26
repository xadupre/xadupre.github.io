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

C ABI contract
--------------

The plugin implements the ORT custom-op C ABI from ``onnxruntime_c_api.h``,
not an execution-provider ABI. Its exported entry point is:

.. code-block:: cpp

   extern "C" OrtStatus *ORT_API_CALL
   RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base);

.. list-table:: Registration and kernel lifecycle
   :header-rows: 1
   :widths: 35 65

   * - API / descriptor
     - Implementation contract
   * - ``RegisterCustomOps``
     - Export with shared-library visibility. Attach the ``onnx_light`` and
       ``onnx_light_cpu`` domains to the supplied session options.
   * - ``GetApi``
     - Request version 18 from the host; do not resolve or link ORT functions.
   * - ``CreateCustomOpDomain``, ``CustomOpDomain_Add``, ``AddCustomOpDomain``
     - Create domains, add operator descriptors and register them with the options.
   * - ``OrtCustomOp``
     - Zero-initialize, set ``version = 18`` and supply name, provider,
       input/output count, concrete type, required-argument, CPU memory and
       start/end opset-version queries.
   * - ``CreateKernelV2``
     - Receive ``OrtApi`` and ``OrtKernelInfo``; return retained state through
       ``void **``. Read LeakyRelu's ``alpha`` with
       ``KernelInfoGetAttribute_float``. Log has no attributes.
   * - ``KernelComputeV2``
     - Receive retained state and the current ``OrtKernelContext``. Use
       ``KernelContext_GetInput``, tensor type/shape/data queries and
       ``KernelContext_GetOutput`` to adapt CPU buffers to native kernels.
   * - ``KernelDestroy``
     - Release retained state when the host is finished.

.. list-table:: Errors and resource ownership
   :header-rows: 1
   :widths: 35 65

   * - Resource / result
     - Ownership contract
   * - Registration, creation and computation
     - Return ``nullptr`` on success or ``OrtStatus *`` on failure.
       Translate exceptions through ``CreateStatus``; never cross the C ABI.
   * - Returned status
     - The caller owns it and calls ``ReleaseStatus``.
   * - Tensor metadata
     - Pair ``GetTensorTypeAndShape`` with ``ReleaseTensorTypeAndShapeInfo``.
   * - Context values and buffers
     - Borrowed from the host. Do not free them or retain invocation-specific
       values or data pointers.
   * - Descriptors and domains
     - Keep alive for all referencing sessions. Cache domains per API table;
       release with ``ReleaseCustomOpDomain`` when the plugin unloads.
       The host and API table must outlive that cleanup.

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
