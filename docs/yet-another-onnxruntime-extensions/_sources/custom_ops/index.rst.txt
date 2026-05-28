Available Custom Ops
====================

This section lists all custom ONNX Runtime operators provided by
*yet-another-onnxruntime-extensions*.  Each family of operators has its own
dedicated page:

.. toctree::
    :maxdepth: 1

    sparse_cpu

.. only:: not ci

    .. toctree::
        :maxdepth: 1

        fused_cuda

Sparse CPU Operators
    Operators registered under ``yaourt.ortops.sparse.cpu`` running on
    the ``CPUExecutionProvider``.  Included as a pre-built binary in the
    wheel — see :doc:`sparse_cpu` for the full catalogue.

.. only:: not ci

    Fused Kernel CUDA Operators
        Operators registered under ``yaourt.ortops.fused_kernel.cuda`` running
        on the ``CUDAExecutionProvider``.  Require a CUDA-enabled build — see
        :doc:`fused_cuda` for the full catalogue and build instructions.
