onnxruntime execution-provider plugins called by onnx-light
===========================================================

Let the standalone onnx-light runtime delegate supported standard ONNX nodes
to trusted ORT-compatible EP plugins. This project is planned, not runnable.

Design
------

``onnx_light_ort::bridge_ep_ort_called_by_light`` is currently an interface target
depending on ``common_ort_to_light`` (described in
:doc:`bridge_ops_ort_called_by_light`). It will host foreign EP factory/device
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

Requirements
------------

The interface target uses the C++20/CMake toolchain and incoming common layer.
A working host still needs EP factory/device APIs, capability selection and
kernel-registry or compiled-partition execution. The existing custom-op
API-18 subset is insufficient for EPs. Plugins must be explicitly selected,
trusted native code; this host must not install, link or load ONNX Runtime.

Python snippet
--------------

This model-only example describes work a future EP plugin could claim.
It requires matching ``onnx_light.onnx`` bindings and does not load a plugin.
There is no EP session API yet: ``CustomOpSessionOrtCalledByLight`` is only
for custom operators, not for this standard-domain model.

.. code-block:: python

   from onnx_light.onnx import TensorProto, checker, helper

   model = helper.make_model(
       helper.make_graph(
           [helper.make_node("Log", ["X"], ["Y"])],
           "ort-ep-candidate",
           [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
           [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
       ),
       opset_imports=[helper.make_opsetid("", 18)],
       ir_version=9,
   )
   checker.check_model(model)
