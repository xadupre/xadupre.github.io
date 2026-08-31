Available Kernels
=================

The table below lists the kernels **provided by this repository**
(``onnx-light-kernel-images``). It does not include the kernels that come with
onnx-light itself. The list is generated automatically at documentation build
time by scanning this repository's own C++ registration sources
(``onnx_light_kernel_images/**/*.cc``), so it always reflects the kernels this
repository actually adds.

.. registered-kernels::

Each kernel is registered for the given operator, ONNX domain and device. The
``ImageDecoder`` kernel decodes encoded image bytestreams into ``(H, W, C)``
``tensor(uint8)`` arrays and supports BMP, TIFF, JPEG, JPEG2000, PNG, WebP and
PNM.
