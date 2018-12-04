
---
author: microsoft
title: 'Metadata'
description: 'ONNX format contains metadata related to how the model was produced. It is useful when the model is deployed to production to keep track of which instance was used at a specific time. Let’s see how to do that with a simple logistic regression model trained with scikit-learn and converted with onnxmltools.'
ms.date: 2018-12-04
---    
    



# Metadata



ONNX format contains metadata related to how the model was produced. It is useful when the model is deployed to production to keep track of which instance was used at a specific time. Let’s see how to do that with a simple logistic regression model trained with *scikit-learn* and converted with *onnxmltools*.

```python
from onnxruntime.datasets import get_example
example = get_example("logreg_iris.onnx")

import onnx
model = onnx.load(example)

print("doc_string={}".format(model.doc_string))
print("domain={}".format(model.domain))
print("ir_version={}".format(model.ir_version))
print("metadata_props={}".format(model.metadata_props))
print("model_version={}".format(model.model_version))
print("producer_name={}".format(model.producer_name))
print("producer_version={}".format(model.producer_version))
```



Out:

```text
doc_string=
domain=onnxml
ir_version=3
metadata_props=[]
model_version=0
producer_name=OnnxMLTools
producer_version=1.2.0.0116
```



With *ONNX Runtime*:

```python
from onnxruntime import InferenceSession
sess = InferenceSession(example)
meta = sess.get_modelmeta()

print("custom_metadata_map={}".format(meta.custom_metadata_map))
print("description={}".format(meta.description))
print("domain={}".format(meta.domain, meta.domain))
print("graph_name={}".format(meta.graph_name))
print("producer_name={}".format(meta.producer_name))
print("version={}".format(meta.version))
```



Out:

```text
custom_metadata_map={}
description=
domain=onnxml
graph_name=3c59201b940f410fa29dc71ea9d5767d
producer_name=OnnxMLTools
version=0
```



**Total running time of the script:** ( 0 minutes  0.004 seconds)
