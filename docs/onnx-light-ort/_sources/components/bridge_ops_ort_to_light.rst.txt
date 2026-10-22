Custom-op bridge: plugins in onnx-light
=======================================

``onnx_light_ort::bridge_ops_ort_to_light`` is currently an interface target
depending on :doc:`common_ort_to_light`. It will load independently supplied
custom-op libraries, invoke ``RegisterCustomOps`` with compatibility API
tables, resolve descriptors and register session-owned kernels in onnx-light.

Ops-specific loading and registration belong here. API tables, opaque values,
tensor ownership and resource lifetime utilities are shared with
:doc:`bridge_ep_ort_to_light` through the common library, never by linking the
two bridges to each other. This direction must not install, load or link ORT.

Runtime implementation and executable tests remain planned. The first loader
must add audited independent fixture libraries, API negotiation, registration
failure, borrowed/owned values and unload-order tests. Existing common-layer
dependency smoke tests are not represented as loader unit coverage.
