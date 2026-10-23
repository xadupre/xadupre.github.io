onnx-light called by onnxruntime through an execution provider
==============================================================

Let ONNX Runtime assign supported standard ONNX nodes to onnx-light, without
rewriting them as custom operators. This project is planned, not runnable.

Design
------

``onnx_light_ort::bridge_ep_light_called_by_ort`` is currently an interface target.
It depends on ``common_light_to_ort``, independently of the custom-op bridge.
The shared layer is described in :doc:`bridge_ops_light_called_by_ort`.

The future ``custom_ep`` library will use this bridge for EP-specific
capability selection, kernel-registry and executable-partition adaptation.
It shares native execution, buffer validation and status conversion through
the common library rather than importing custom-op callbacks or its attribute
contract. EP kernel factories must define their own standard-node defaults.

Runtime implementation and executable tests remain planned. The first
implementation must add factory/kernel lifecycle, supported/unsupported node
selection and real-ORT placement tests; an interface target alone is not
evidence that an EP works.

Requirements
------------

The current interface target uses the same C++20/CMake toolchain and outgoing
common layer as the custom-op project. A working EP still requires the
``custom_ep`` entry-point library, factory/device ownership, capability
selection and executable kernels, plus a real ORT host supporting the pinned
EP API. None of those EP runtime paths is implemented yet.

Python snippet
--------------

This model-only example illustrates a standard ONNX node a future EP could
claim. It requires matching ``onnx_light.onnx`` bindings, not an installed ORT.
It does **not** register or execute an EP; unsupported nodes must remain with
the ORT host when the provider is implemented.

.. code-block:: python

   from onnx_light.onnx import TensorProto, checker, helper

   model = helper.make_model(
       helper.make_graph(
           [helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.125)],
           "light-ep-candidate",
           [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
           [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
       ),
       opset_imports=[helper.make_opsetid("", 18)],
       ir_version=9,
   )
   checker.check_model(model)

.. include:: custom_ep.inc
