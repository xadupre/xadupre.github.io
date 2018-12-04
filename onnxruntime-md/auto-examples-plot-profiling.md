
---
author: microsoft
title: 'Profile the execution of a simple model'
description: 'ONNX Runtime can profile the execution of the model. This example shows how to interpret the results.'
ms.date: 2018-12-04
---    
    



# Profile the execution of a simple model



*ONNX Runtime* can profile the execution of the model. This example shows how to interpret the results.

```python
import onnxruntime as rt
import numpy
from onnxruntime.datasets import get_example
```



Let’s load a very simple model and compute some prediction.

```python
example1 = get_example("mul_1.pb")
sess = rt.InferenceSession(example1)
input_name = sess.get_inputs()[0].name

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
res = sess.run(None, {input_name: x})
print(res)
```



Out:

```python
[array([[ 1.,  4.],
       [ 9., 16.],
       [25., 36.]], dtype=float32)]
```



We need to enable to profiling before running the predictions.

```python
options = rt.SessionOptions()
options.enable_profiling = True
sess_profile = rt.InferenceSession(example1, options)
input_name = sess.get_inputs()[0].name

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)

sess.run(None, {input_name: x})
prof_file = sess_profile.end_profiling()
print(prof_file)
```



Out:

```text
onnxruntime_profile__2018-12-04_11-57-44.json
```



The results are stored un a file in JSON format. Let’s see what it contains.

```python
import json
with open(prof_file, "r") as f:
    sess_time = json.load(f)
import pprint
pprint.pprint(sess_time)
```



Out:

```python
[{'args': {},
  'cat': 'Session',
  'dur': 215,
  'name': 'model_loading_uri',
  'ph': 'X',
  'pid': 19888,
  'tid': 50744,
  'ts': 16},
 {'args': {},
  'cat': 'Session',
  'dur': 111,
  'name': 'session_initialization',
  'ph': 'X',
  'pid': 19888,
  'tid': 50744,
  'ts': 256}]
```



**Total running time of the script:** ( 0 minutes  0.017 seconds)
