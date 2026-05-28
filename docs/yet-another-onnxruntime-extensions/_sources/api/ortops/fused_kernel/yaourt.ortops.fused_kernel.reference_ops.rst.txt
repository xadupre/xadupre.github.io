yaourt.ortops.fused\_kernel.reference\_ops
==========================================

.. automodule:: yaourt.ortops.fused_kernel.reference_ops
    :members:
    :special-members:

Usage example
-------------

.. runpython::
    :showcode:

    import numpy as np
    import onnx.helper as oh
    import onnx
    from yaourt.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    DOMAIN = "yaourt.ortops.fused_kernel.cuda"
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("MulMul", ["A", "B", "C"], ["Z"], domain=DOMAIN)],
            "mulmul_graph",
            [oh.make_tensor_value_info(n, TFLOAT, [None]) for n in "ABC"],
            [oh.make_tensor_value_info("Z", TFLOAT, [None])],
        ),
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(DOMAIN, 1)],
        ir_version=10,
    )
    ref = ExtendedReferenceEvaluator(model)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    (result,) = ref.run(None, {"A": a, "B": b, "C": c})
    print(result)
