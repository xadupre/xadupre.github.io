API Reference
=============

C++ API
-------

.. doxygenfunction:: onnx_light_kernel_images::RegisterImageKernels
   :project: onnx_light_kernel_images

Python API
----------

.. py:module:: onnx_light_kernel_images.onnx_py._imgpykernels

.. py:function:: register_image_kernels()

   Registers the ``ImageDecoder`` kernel (ai.onnx domain) with the onnx-light
   kernel dispatch table.

   Supported formats: BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM. JPEG2000 and
   WebP are decoded through the ``libopenjp2`` and ``libwebp`` shared
   libraries, loaded dynamically at runtime.

   Idempotent: calling more than once is safe and cheap.

.. py:function:: has_image_kernels() -> bool

   Returns ``True`` when the image kernel extension is available.
