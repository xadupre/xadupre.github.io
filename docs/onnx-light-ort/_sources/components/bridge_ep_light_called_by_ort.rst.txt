onnx-light called by onnxruntime through an execution provider
==============================================================

``onnx_light_ort::bridge_ep_light_called_by_ort`` is currently an interface target.
It depends on :doc:`common_light_to_ort`, independently of
:doc:`bridge_ops_light_called_by_ort`.

The future :doc:`custom_ep` library will use this bridge for EP-specific
capability selection, kernel-registry and executable-partition adaptation.
It shares native execution, buffer validation and status conversion through
the common library rather than importing custom-op callbacks or its attribute
contract. EP kernel factories must define their own standard-node defaults.

Runtime implementation and executable tests remain planned. The first
implementation must add factory/kernel lifecycle, supported/unsupported node
selection and real-ORT placement tests; an interface target alone is not
evidence that an EP works.
