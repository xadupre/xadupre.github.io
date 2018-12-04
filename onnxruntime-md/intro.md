
---
author: microsoft
title: 'Python Bindings for ONNX Runtime'
description: 'ONNX Runtime enables high-performance evaluation of trained machine learning (ML) models while keeping resource usage low. Building on Microsoft’s dedication to the Open Neural Network Exchange (ONNX) community, it supports traditional ML models as well as Deep Learning algorithms in the ONNX-ML format.'
ms.date: 2018-12-04
---    
    



# Python Bindings for ONNX Runtime



ONNX Runtime enables high-performance evaluation of trained machine learning (ML) models while keeping resource usage low. Building on Microsoft’s dedication to the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) community, it supports traditional ML models as well as Deep Learning algorithms in the [ONNX-ML format](https://github.com/onnx/onnx/blob/master/docs/IR.md).

Contents:

* [Tutorial](tutorial.md)
* [API Summary](api-summary.md)
* [Gallery of examples](examples-md.md)




The core library is implemented in C++. *ONNX Runtime* is available on PyPi for Linux Ubuntu 16.04, Python 3.5+ for both [CPU](https://pypi.org/project/onnxruntime/) and [GPU](https://pypi.org/project/onnxruntime-gpu/). This example demonstrates a simple prediction for an [ONNX-ML format](https://github.com/onnx/onnx/blob/master/docs/IR.md) model. The following file ``model.onnx`` is taken from github [onnx...test_sigmoid](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node/test_sigmoid).

```python
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
X = numpy.random.random((3,4,5)).astype(numpy.float32)
res = sess.run([output_name], {input_name: x})
pred_onnx = sess.run(None, {input_name: X})
```
