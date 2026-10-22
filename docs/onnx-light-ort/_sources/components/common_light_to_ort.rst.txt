Shared support for onnx-light in ORT
=====================================

``onnx_light_ort/common_light_to_ort/`` builds the static target
``onnx_light_ort::common_light_to_ort``. It adapts native onnx-light execution
to an ORT host's current tensors and kernel context. Both
:doc:`bridge_ops_light_called_by_ort` and :doc:`bridge_ep_light_called_by_ort` depend on it.
It contains no ops/EP registration, descriptor or factory ownership.

The current ``LeakyReluKernel`` retains one native kernel and its alpha
attribute. Each ``Compute`` call validates CPU memory, float32 type, rank,
dimensions, checked element/byte counts, non-null nonempty buffers, alignment
and non-overlap before borrowing payloads. It owns neither the input/output
storage nor the ORT context and retains none of them between calls.
``GetUnaryFloat32Buffers`` shares these checks with ``LogKernel``, which calls
onnx-light-cpu's ``ml_cpu_compute::LogFloat32`` directly into the host's output
buffer. The numerical dependency remains private to this common library;
neither the incoming common library nor ONNX Runtime is linked.
Native exceptions propagate to the bridge's C ABI boundary; ``api_status.h``
provides the shared exception-to-status conversion used at those boundaries
and by the entry-point library.
Actual allocation capacity remains part of the real host's valid tensor
contract; see :doc:`../architecture`.

Unit tests
----------

.. code-block:: bash

   ctest --test-dir build -L common_light_to_ort --output-on-failure
   ctest --test-dir build -R common_light_to_ort_unit --output-on-failure

The component's ``tests/test_leaky_relu_kernel.cc`` exercises the real adapter
and numerical kernel using a minimal explicit C API test host. It covers
repeated vector/matrix/scalar/empty/rank-16 execution, input preservation,
missing inputs/outputs, device/type/rank rejection, negative dimensions,
element and byte-count overflow, null or overlapping buffers, unchanged
status propagation, metadata cleanup and recovery after errors.

These native tests need no ORT installation or mocking framework. The
fixture can supply invalid metadata that real ORT usually rejects before
dispatch. The separately enabled real-ORT suite also checks caller-provided
aligned/unaligned buffers and direct IO-bound outputs.
