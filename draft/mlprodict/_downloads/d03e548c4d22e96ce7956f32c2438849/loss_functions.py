#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Loss function in ONNX
# 
# The following notebook show how to translate common loss function into ONNX.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


from mlprodict.plotting.text_plot import onnx_simple_text_plot
get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## Square loss
# 
# The first example shows how to use [onnx](https://github.com/onnx/onnx) API to represent the square loss function $E(X,Y) = \sum_i(x_i-y_i)^2$ where $X=(x_i)$ and $Y=(y_i)$.

# ### numpy function

# In[3]:


import numpy


def square_loss(X, Y):
    return numpy.sum((X - Y) ** 2, keepdims=1)


x = numpy.array([0, 1, 2], dtype=numpy.float32)
y = numpy.array([0.5, 1, 2.5], dtype=numpy.float32)
square_loss(x, y)


# ### onnx version
# 
# Following example is based on [onnx Python API](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md), described with more detailed at [Introduction to onnx Python API](http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/tutorials/tutorial_onnx/python.html).

# In[4]:


from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info
from onnx import TensorProto

nodes = [make_node('Sub', ['X', 'Y'], ['diff']),
         make_node('Mul', ['diff', 'diff'], ['diff2']),
         make_node('ReduceSum', ['diff2'], ['loss'])]

graph = make_graph(nodes, 'square_loss',
                   [make_tensor_value_info('X', TensorProto.FLOAT, [None]),
                    make_tensor_value_info('Y', TensorProto.FLOAT, [None])],
                   [make_tensor_value_info('loss', TensorProto.FLOAT, [None])])
model = make_model(graph)
del model.opset_import[:]
opset = model.opset_import.add()
opset.domain = ''
opset.version = 14


# In[5]:


print(onnx_simple_text_plot(model))


# In[6]:


get_ipython().run_line_magic('onnxview', 'model')


# Let's check it gives the same results.

# In[7]:


from onnxruntime import InferenceSession
sess = InferenceSession(model.SerializeToString())
sess.run(None, {'X': x, 'Y': y})


# ### second API from sklearn-onnx
# 
# The previous API is quite verbose. [sklearn-onnx](https://onnx.ai/sklearn-onnx/) implements a more simple API to do it where every onnx operator is made available as a class. It was developped to speed up the implementation of converters for scikit-learn (see [sklearn-onnx](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_icustom_converter.html)).

# In[8]:


from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxMul, OnnxReduceSum

diff = OnnxSub('X', 'Y')
nodes = OnnxReduceSum(OnnxMul(diff, diff))
model = nodes.to_onnx({'X': x, 'Y': y})

print(onnx_simple_text_plot(model))


# In[9]:


sess = InferenceSession(model.SerializeToString())
sess.run(None, {'X': x, 'Y': y})


# As the previous example, this function only allows float32 arrays. It fails for any other type.

# In[10]:


try:
    sess.run(None, {'X': x.astype(numpy.float64), 
                    'Y': y.astype(numpy.float64)})
except Exception as e:
    print(e)


# ### numpy API
# 
# Second example is much more simple than the first one but it requires to know [ONNX operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md). The most difficult type is about writing the signature. In the following example, it take two arrays of the same type `T` and returns an array of the same type, `T` being any element type (float32, float64, int64, ...).

# In[11]:


from mlprodict.npy import onnxnumpy_np, NDArrayType
import mlprodict.npy.numpy_onnx_impl as npnx

@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)))
def onnx_square_loss(X, Y):
    return npnx.sum((X - Y) ** 2, keepdims=1)

onnx_square_loss(x, y)


# This API compiles an ONNX graphs for every element type. So it works float64 as well.

# In[12]:


onnx_square_loss(x.astype(numpy.float64), y.astype(numpy.float64))


# That's why method `to_onnx` requires to specify the element type before the method can return the associated ONNX graph.

# In[13]:


onx = onnx_square_loss.to_onnx(key=numpy.float64)
print(onnx_simple_text_plot(onx))


# ## log loss
# 
# The log loss is defined as the following: $L(y, s) = (1 - y)\log(1 - p(s)) + y \log(p(s))$ where $p(s) = sigmoid(s) = \frac{1}{1 + \exp(-s)}$. Let's start with the numpy version.

# ###  numpy function

# In[14]:


from scipy.special import expit

def log_loss(y, s):
    ps = expit(-s)
    ls = (1 - y) * numpy.log(1 - ps) + y * numpy.log(ps)
    return numpy.sum(ls, keepdims=1)

y = numpy.array([0, 1, 0, 1], dtype=numpy.float32)
s = numpy.array([1e-50, 1e50, 0, 1], dtype=numpy.float32)
log_loss(y, s)


# The function may return unexpected values because `log(0)` does not exist. The trick is usually to clip the value.

# In[15]:


def log_loss_clipped(y, s, eps=1e-6):
    ps = numpy.clip(expit(-s), eps, 1-eps)
    ls = (1 - y) * numpy.log(1 - ps) + y * numpy.log(ps)
    return numpy.sum(ls, keepdims=1)

log_loss_clipped(y, s)


# ### numpy to onnx with onnx operators

# In[16]:


from skl2onnx.algebra.onnx_ops import (
    OnnxClip, OnnxSigmoid, OnnxLog, OnnxAdd, OnnxSub, OnnxMul, OnnxNeg)

eps = numpy.array([1e-6], dtype=numpy.float32)
one = numpy.array([1], dtype=numpy.float32)

ps = OnnxClip(OnnxSigmoid(OnnxNeg('S')), eps, 1-eps)
ls1 = OnnxMul(OnnxSub(one, 'Y'), OnnxLog(OnnxSub(one, ps)))
ls2 = OnnxMul('Y', OnnxLog(ps))
nodes = OnnxReduceSum(OnnxAdd(ls1, ls2), keepdims=1)
model = nodes.to_onnx({'Y': y, 'S': s})

print(onnx_simple_text_plot(model))


# In[17]:


get_ipython().run_line_magic('onnxview', 'model')


# In[18]:


sess = InferenceSession(model.SerializeToString())
sess.run(None, {'Y': y, 'S': s})


# Same results.

# ### Back to onnx API
# 
# Coding the previous graph would take too much time but it is still possible to build it from the ONNX graph we just got.

# In[19]:


from mlprodict.onnx_tools.onnx_export import export2onnx
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
print(export2onnx(onnx_rename_names(model)))


# ### numpy to onnx with numpy API

# In[20]:


@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)),
              op_version=15)
def onnx_log_loss(y, s, eps=1e-6):

    one = numpy.array([1], dtype=s.dtype)
    ceps = numpy.array([eps], dtype=s.dtype)
    
    ps = npnx.clip(npnx.expit(-s), ceps, one-ceps)
    ls = (one - y) * npnx.log(one - ps) + y * npnx.log(ps)
    return npnx.sum(ls, keepdims=1)

onnx_log_loss(y, s, eps=1e-6)


# In[21]:


onnx_log_loss(y, s, eps=1e-4)


# The implementation is slightly different from the numpy implementation. `1 - y` cannot be used because 1 is an integer and the function needs to know if it is a integer 32 or 64. `numpy.array([1], dtype=s.dtype) - y` is better in this case to avoid any ambiguity on the type of constant `1`. That may be revisited in the future. The named argument is part of the ONNX graph as an initializer. An new graph is generated every time the function sees a new value. That explains why the following instructions cannot return one ONNX graph as they are more than one:

# In[22]:


try:
    onnx_log_loss.to_onnx(key=numpy.float32)
except Exception as e:
    print(e)


# Let's see the list of available graphs:

# In[23]:


list(onnx_log_loss.signed_compiled)


# Let's pick the first one.

# In[24]:


from mlprodict.npy import FctVersion
onx = onnx_log_loss.to_onnx(key=FctVersion((numpy.float32,numpy.float32), (1e-06,)))


# In[25]:


print(onnx_simple_text_plot(onx))


# ### no loss but lagg, something difficult to write with onnx

# In[26]:


@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", ), dtypes_out=('T',)))
def lagged(x, lag=2):
    return x[lag:] - x[:-lag]

x = numpy.array([[0, 1], [2, 3], [4, 5], [10, 21]], dtype=numpy.float32)
lagged(x)


# In[27]:


print(onnx_simple_text_plot(lagged.to_onnx(key=numpy.float32)))


# In[28]:


get_ipython().run_line_magic('onnxview', 'lagged.to_onnx(key=numpy.float32)')


# In[29]: