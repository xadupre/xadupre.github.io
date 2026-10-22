Shared support for hosting ORT-compatible plugins
==================================================

``onnx_light_ort::common_ort_to_light`` is a static library exposing the
initial audited CPU custom-op subset of ``OrtApi`` over onnx-light tensors.
It supplies opaque adapters for statuses, registration domains, kernel
attributes, values and contexts. :doc:`bridge_ops_ort_called_by_light` owns loading,
descriptor selection and the session/kernel/library lifecycle.
The EP bridge remains unimplemented.

``OrtApiBase::GetApi`` accepts only version 18. The full pinned ``OrtApi``
layout is preserved; unsupported slots remain null. Supported entries are:

* ``CreateStatus``, ``GetErrorCode``, ``GetErrorMessage``, ``ReleaseStatus``.
* ``CreateCustomOpDomain``, ``CustomOpDomain_Add``, ``AddCustomOpDomain``,
  ``ReleaseCustomOpDomain``.
* ``KernelInfoGetAttribute_float``.
* ``KernelContext_GetInputCount``, ``KernelContext_GetOutputCount``,
  ``KernelContext_GetInput``, ``KernelContext_GetOutput``.
* ``GetTensorTypeAndShape``, ``ReleaseTensorTypeAndShapeInfo``,
  ``GetTensorElementType``, ``GetDimensionsCount``, ``GetDimensions``,
  ``GetTensorShapeElementCount``, ``GetTensorMutableData``.
* ``GetTensorMemoryInfo``, ``MemoryInfoGetDeviceType``.

These functions implement only dense float32 CPU values. Shape and byte counts
are checked before allocating outputs. The component is independent of the
outgoing common library and never loads ORT. ``tests/ort_test_host.h`` remains
separate test infrastructure for outgoing callback units, not this host.

The dependency-contract smoke test is retained alongside the real incoming
fixture tests. Select both using the ``common_ort_to_light`` CTest label.
