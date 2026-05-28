Sparse CPU Custom Ops
=====================

This page lists all custom ONNX Runtime operators in the sparse CPU family
provided by *yet-another-onnxruntime-extensions*.  The catalogue is generated
dynamically at documentation-build time by parsing the C++ source files, so
it always reflects the actual implementation without any manual maintenance.

These operators are registered under the ``yaourt.ortops.sparse.cpu`` domain
and run on the ``CPUExecutionProvider``.  The pre-built shared library is
shipped inside the wheel and can be loaded via
:data:`~yaourt.ortops.SPARSE_CPU2_LIB_PATH`.

Operators
---------

.. runpython::
    :rst:

    from yaourt.ortops.doc import print_cpu_ops_rst
    print_cpu_ops_rst()
