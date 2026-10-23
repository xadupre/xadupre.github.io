onnx-light-ort
==============

.. image:: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml/badge.svg?branch=main
   :target: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml
   :alt: Documentation build status

.. image:: https://codecov.io/gh/xadupre/onnx-light-ort/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xadupre/onnx-light-ort
   :alt: Native C++ code coverage

Interoperability between onnx-light and ONNX Runtime custom operators and
execution providers, in both directions.

.. raw:: html

   <object class="project-diagram" type="image/svg+xml" data="_static/projects.svg"
           aria-label="Four projects: select a custom-operator or execution-provider bridge">
     <ul>
       <li><a href="components/bridge_ops_light_called_by_ort.html">onnx-light custom operators called by ORT</a></li>
       <li><a href="components/bridge_ops_ort_called_by_light.html">ORT custom operators called by onnx-light</a></li>
       <li><a href="components/bridge_ep_light_called_by_ort.html">onnx-light execution provider called by ORT (planned)</a></li>
       <li><a href="components/bridge_ep_ort_called_by_light.html">ORT execution providers called by onnx-light (planned)</a></li>
     </ul>
   </object>

.. only:: not html

   .. image:: _static/projects.svg
      :alt: Four bridges: custom operators (implemented) and execution providers (planned), in both directions between onnx-light and ONNX Runtime.
      :width: 100%

.. toctree::
   :maxdepth: 1

   introduction
   python_api
   components/index
   auto_examples/index
   newt_steps/index
