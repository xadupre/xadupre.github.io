Introduction
============

onnx-light-ort connects onnx-light and ONNX Runtime (ORT) through native
custom-operator interfaces in both directions:

* Expose onnx-light kernels as custom operators executed by a real ORT host.
* Execute audited ORT custom-op libraries inside standalone onnx-light,
  without loading or linking ORT.

Custom ops currently execute in both directions: native onnx-light
``LeakyRelu`` in real ORT, and API-18 CPU custom-op libraries in onnx-light
through a limited single-node float32 host. Execution providers remain planned.

The :doc:`projects section <components/index>` groups one page per project,
including its responsibility, dependencies and implementation status.
The :doc:`examples gallery <auto_examples/index>` demonstrates both working
custom-op directions. The :doc:`next steps <newt_steps/index>` describe the
remaining interoperability roadmap.

.. toctree::
   :maxdepth: 1

   architecture
   development
