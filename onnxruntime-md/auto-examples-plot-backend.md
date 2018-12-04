
---
author: microsoft
title: 'ONNX Runtime Backend for ONNX'
description: 'ONNX Runtime extends the onnx backend API to run predictions using this runtime. Let’s use the API to compute the prediction of a simple logistic regression model.'
ms.date: 2018-12-04
---    
    



# ONNX Runtime Backend for ONNX



*ONNX Runtime* extends the [onnx backend API](https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md) to run predictions using this runtime. Let’s use the API to compute the prediction of a simple logistic regression model.

```python
import numpy as np
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnx import load

name = datasets.get_example("logreg_iris.onnx")
model = load(name)

rep = backend.prepare(model, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))
```



Out:

```python
label=[1]
probabilities=[{0: 0.02731134556233883, 1: 0.5175684094429016, 2: 0.4551202654838562}]
```



The device depends on how the package was compiled, GPU or CPU.

```python
from onnxruntime import get_device
print(get_device())
```



Out:

```python
CPU-MKL-DNN
```



The backend can also directly load the model without using *onnx*.

```python
rep = backend.prepare(name, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))
```



Out:

```python
label=[1]
probabilities=[{0: 0.02731134556233883, 1: 0.5175684094429016, 2: 0.4551202654838562}]
```



The backend API is implemented by other frameworks and makes it easier to switch between multiple runtimes with the same API.



**Total running time of the script:** ( 0 minutes  0.100 seconds)
