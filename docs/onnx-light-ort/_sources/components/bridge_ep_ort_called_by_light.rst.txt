EP bridge: ORT plugins called by onnx-light
===========================================

``onnx_light_ort::bridge_ep_ort_called_by_light`` is currently an interface target
depending on :doc:`common_ort_to_light`. It will host foreign EP factory/device
callbacks, provider selection and kernel-registry or compiled-partition
execution in onnx-light.

EP-specific registration belongs here. API tables, opaque values, tensor
ownership and resource lifetime utilities are shared with
:doc:`bridge_ops_ort_called_by_light` through the common library, never by linking the
two bridges to each other. This direction must not install, load or link ORT.

Runtime implementation and executable tests remain planned. The first
implementation must add factory/device ownership, partial failure, capability
selection, kernel/compiled-state reuse and teardown tests using independent
fixture EPs. Common-layer dependency smoke tests do not validate EP hosting.
