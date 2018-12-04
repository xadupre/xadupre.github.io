
---
author: microsoft
title: 'Train, convert and predict with ONNX Runtime'
description: 'This example demonstrates an end to end scenario starting with the training of a scikit-learn pipeline which takes as inputs not a regular vector but a dictionary { int: float } as its first step is a DictVectorizer.'
ms.date: 2018-12-04
---    
    



# Train, convert and predict with ONNX Runtime



This example demonstrates an end to end scenario starting with the training of a scikit-learn pipeline which takes as inputs not a regular vector but a dictionary ``{ int: float }`` as its first step is a [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html).

* [Train a pipeline](#train-a-pipeline)
* [Conversion to ONNX format](#conversion-to-onnx-format)



## Train a pipeline



The first step consists in retrieving the boston datset.

```python
import pandas
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pandas.DataFrame(X_train[:,1:]).T.to_dict().values()
X_test_dict = pandas.DataFrame(X_test[:,1:]).T.to_dict().values()
```



We create a pipeline.

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
pipe = make_pipeline(
            DictVectorizer(sparse=False),
            GradientBoostingRegressor())

pipe.fit(X_train_dict, y_train)
```



We compute the prediction on the test set and we show the confusion matrix.

```python
from sklearn.metrics import r2_score

pred = pipe.predict(X_test_dict)
print(r2_score(y_test, pred))
```



Out:

```python
0.872098032796385
```


## Conversion to ONNX format



We use module [onnxmltools](https://github.com/onnx/onnxmltools) to convert the model into ONNX format.

```python
from onnxmltools import convert_sklearn
from onnxmltools.utils import save_model
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, DictionaryType, SequenceType

# initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
onx = convert_sklearn(pipe, initial_types=initial_type)
save_model(onx, "pipeline_vectorize.onnx")
```



We load the model with ONNX Runtime and look at its input and output.

```python
import onnxruntime as rt
sess = rt.InferenceSession("pipeline_vectorize.onnx")

import numpy
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print("input name='{}' and shape={} and type={}".format(inp.name, inp.shape, inp.type))
print("output name='{}' and shape={} and type={}".format(out.name, out.shape, out.type))
```



Out:

```text
input name='float_input' and shape=[] and type=map(int64,tensor(float))
output name='variable1' and shape=[1, 1] and type=tensor(float)
```



We compute the predictions. We could do that in one call:

```python
try:
    pred_onx = sess.run([out.name], {inp.name: X_test_dict})[0]
except RuntimeError as e:
    print(e)
```



Out:

```text
Method run failed due to: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (class onnxruntime::SequenceType<class std::vector<class std::map<__int64,float,struct std::less<__int64>,class std::allocator<struct std::pair<__int64 const ,float> > >,class std::allocator<class std::map<__int64,float,struct std::less<__int64>,class std::allocator<struct std::pair<__int64 const ,float> > > > > >) , expected: (class onnxruntime::MapType<class std::map<__int64,float,struct std::less<__int64>,class std::allocator<struct std::pair<__int64 const ,float> > > >)
```



But it fails because, in case of a DictVectorizer, ONNX Runtime expects one observation at a time.

```python
pred_onx = [sess.run([out.name], {inp.name: row})[0][0, 0] for row in X_test_dict]
```



We compare them to the model’s ones.

```python
print(r2_score(pred, pred_onx))
```



Out:

```python
0.9999928931389793
```



Very similar. *ONNX Runtime* uses floats instead of doubles, that explains the small discrepencies.



**Total running time of the script:** ( 0 minutes  3.306 seconds)
