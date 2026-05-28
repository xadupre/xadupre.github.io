C++ API Reference
=================

This section documents the public C++ API of
*yet-another-onnxruntime-extensions*, generated from the Doxygen XML output.

Custom ORT Kernel Structs
--------------------------

Kernel structs declared in ``yaourt/ortops/sparse/cpu/ort_sparse_lite.h``.

.. doxygenfile:: ort_sparse_lite.h
    :project: yaourt

Sparse Tensor Encoding
-----------------------

The ``sparse_struct`` type used to encode sparse tensors in-memory, declared
in ``yaourt/cpp/include/common/sparse_tensor.h``.

.. doxygenfile:: sparse_tensor.h
    :project: yaourt

Helper Utilities
-----------------

String-building and enforcement helpers declared in
``yaourt/cpp/include/yaourt_helpers.h``.

.. doxygenfile:: yaourt_helpers.h
    :project: yaourt
