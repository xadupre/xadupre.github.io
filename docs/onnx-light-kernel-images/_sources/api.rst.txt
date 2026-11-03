API Reference
=============

C++ API
-------

.. doxygenfunction:: onnx_light_kernel_images::RegisterImageKernels()
   :project: onnx_light_kernel_images

.. doxygenfunction:: onnx_light_kernel_images::RegisterImageKernels(onnx_light::core::runtime::RuntimeContext &)
   :project: onnx_light_kernel_images

Python API
----------

.. py:module:: onnx_light_kernel_images.onnx_py._imgpykernels

.. py:function:: register_image_kernels(context=None)

   Registers the ``ImageDecoder`` kernel (ai.onnx domain) with the onnx-light
   kernel dispatch table.

   Pass a ``ReferenceEvaluator`` or low-level ``RuntimeContext`` to register
   only for that context and its subgraphs, without changing the global table
   or other evaluators. Register before the first run; already-resolved kernels
   are not replaced. Omitting ``context`` or passing ``None`` registers globally.

   Supported formats: BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM. JPEG2000 and
   WebP are decoded through the ``libopenjp2`` and ``libwebp`` shared
   libraries, loaded dynamically at runtime.

   Idempotent: calling more than once is safe and cheap.

.. py:function:: has_image_kernels() -> bool

   Returns ``True`` when the image kernel extension is available.
