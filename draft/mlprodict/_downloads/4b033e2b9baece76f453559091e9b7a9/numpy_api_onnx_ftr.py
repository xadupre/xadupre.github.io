#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Introduction to a numpy API for ONNX: FunctionTransformer
# 
# This notebook shows how to write python functions similar functions as numpy offers and get a function which can be converted into ONNX.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## A pipeline with FunctionTransformer

# In[3]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[4]:


import numpy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
            FunctionTransformer(numpy.log),
            StandardScaler(),
            LogisticRegression())
pipe.fit(X_train, y_train)


# Let's convert it into ONNX.

# In[5]:


from mlprodict.onnx_conv import to_onnx
try:
    onx = to_onnx(pipe, X_train.astype(numpy.float64))
except (RuntimeError, TypeError) as e:
    print(e)


# ## Use ONNX instead of numpy
# 
# The pipeline cannot be converter because the converter does not know how to convert the function (`numpy.log`) held by `FunctionTransformer` into ONNX. One way to avoid that is to replace it by a function `log` defined with *ONNX* operators and executed with an ONNX runtime.

# In[6]:


import mlprodict.npy.numpy_onnx_pyrt as npnxrt

pipe = make_pipeline(
            FunctionTransformer(npnxrt.log),
            StandardScaler(),
            LogisticRegression())
pipe.fit(X_train, y_train)


# In[7]:


onx = to_onnx(pipe, X_train.astype(numpy.float64), rewrite_ops=True)


# In[8]:


get_ipython().run_line_magic('onnxview', 'onx')


# The operator `Log` is belongs to the graph. There is some overhead by using this function on small matrices. The gap is much less on big matrices.

# In[9]:


get_ipython().run_line_magic('timeit', 'numpy.log(X_train)')


# In[10]:


get_ipython().run_line_magic('timeit', 'npnxrt.log(X_train)')


# ## Slightly more complex functions with a FunctionTransformer
# 
# What about more complex functions? It is a bit more complicated too. The previous syntax does not work.

# In[11]:


def custom_fct(x):
    return npnxrt.log(x + 1)

pipe = make_pipeline(
            FunctionTransformer(custom_fct),
            StandardScaler(),
            LogisticRegression())
pipe.fit(X_train, y_train)


# In[12]:


try:
    onx = to_onnx(pipe, X_train.astype(numpy.float64), rewrite_ops=True)
except TypeError as e:
    print(e)


# The syntax is different.

# In[13]:


from typing import Any
from mlprodict.npy import onnxnumpy_default, NDArray
import mlprodict.npy.numpy_onnx_impl as npnx

@onnxnumpy_default
def custom_fct(x: NDArray[(None, None), numpy.float64]) -> NDArray[(None, None), numpy.float64]:
    return npnx.log(x + numpy.float64(1))

pipe = make_pipeline(
            FunctionTransformer(custom_fct),
            StandardScaler(),
            LogisticRegression())
pipe.fit(X_train, y_train)


# In[14]:


onx = to_onnx(pipe, X_train.astype(numpy.float64), rewrite_ops=True)
get_ipython().run_line_magic('onnxview', 'onx')


# Let's compare the time to *numpy*.

# In[15]:


def custom_numpy_fct(x):
    return numpy.log(x + numpy.float64(1))

get_ipython().run_line_magic('timeit', 'custom_numpy_fct(X_train)')


# In[16]:


get_ipython().run_line_magic('timeit', 'custom_fct(X_train)')


# The new function is slower but the gap is much less on bigger matrices. The default ONNX runtime has a significant cost compare to the cost of a couple of operations on small matrices.

# In[17]:


bigx = numpy.random.rand(10000, X_train.shape[1])
get_ipython().run_line_magic('timeit', 'custom_numpy_fct(bigx)')


# In[18]:


get_ipython().run_line_magic('timeit', 'custom_fct(bigx)')


# ## Function transformer with FFT
# 
# The following function is equivalent to the module of the output of a FFT transform. The matrix $M_{kn}$ is defined by $M_{kn}=(\exp(-2i\pi kn/N))_{kn}$. Complex features are then obtained by computing $MX$. Taking the module leads to real features: $\sqrt{Re(MX)^2 + Im(MX)^2}$. That's what the following function does.

# ###  numpy implementation

# In[19]:


def custom_fft_abs_py(x):
    "onnx fft + abs python"
    # see https://jakevdp.github.io/blog/
    # 2013/08/28/understanding-the-fft/
    dim = x.shape[1]
    n = numpy.arange(dim)
    k = n.reshape((-1, 1)).astype(numpy.float64)
    kn = k * n * (-numpy.pi * 2 / dim)
    kn_cos = numpy.cos(kn)
    kn_sin = numpy.sin(kn)
    ekn = numpy.empty((2,) + kn.shape, dtype=x.dtype)
    ekn[0, :, :] = kn_cos
    ekn[1, :, :] = kn_sin
    res = numpy.dot(ekn, x.T)
    tr = res ** 2
    mod = tr[0, :, :] + tr[1, :, :]
    return numpy.sqrt(mod).T

x = numpy.random.randn(3, 4).astype(numpy.float32)
custom_fft_abs_py(x)


# ### ONNX implementation

# This function cannot be exported into ONNX unless it is written with ONNX operators. This is where the numpy API for ONNX helps speeding up the process.

# In[20]:


from mlprodict.npy import onnxnumpy_default, onnxnumpy_np, NDArray
import mlprodict.npy.numpy_onnx_impl as nxnp


def _custom_fft_abs(x):
    dim = x.shape[1]
    n = nxnp.arange(0, dim).astype(numpy.float32)
    k = n.reshape((-1, 1))
    kn = (k * (n * numpy.float32(-numpy.pi * 2))) / dim.astype(numpy.float32)
    kn3 = nxnp.expand_dims(kn, 0)
    kn_cos = nxnp.cos(kn3)
    kn_sin = nxnp.sin(kn3)
    ekn = nxnp.vstack(kn_cos, kn_sin)
    res = nxnp.dot(ekn, x.T)
    tr = res ** 2
    mod = tr[0, :, :] + tr[1, :, :]
    return nxnp.sqrt(mod).T


@onnxnumpy_default
def custom_fft_abs(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx fft + abs"
    return _custom_fft_abs(x)


custom_fft_abs(x)


# `custom_fft_abs` is not a function a class holding an ONNX graph. A method `__call__` executes the ONNX graph with a python runtime.

# In[21]:


fonx = custom_fft_abs.to_onnx()
get_ipython().run_line_magic('onnxview', 'fonx')


# Every intermediate output can be logged.

# In[22]:


custom_fft_abs(x, verbose=1, fLOG=print)


# In[23]:


get_ipython().run_line_magic('timeit', 'custom_fft_abs_py(x)')


# In[24]:


get_ipython().run_line_magic('timeit', 'custom_fft_abs(x)')


# Again the gap is less on bigger matrices. It cannot be faster with the default runtime as it is also using *numpy*. That's another story with *onnxruntime* (see below).

# In[25]:


bigx = numpy.random.randn(10000, x.shape[1]).astype(numpy.float32)
get_ipython().run_line_magic('timeit', 'custom_fft_abs_py(bigx)')


# In[26]:


get_ipython().run_line_magic('timeit', 'custom_fft_abs(bigx)')


# ### Using onnxruntime
# 
# The python runtime is using numpy but is usually quite slow as the runtime needs to go through the graph structure.
# *onnxruntime* is faster.

# In[27]:


@onnxnumpy_np(runtime='onnxruntime')
def custom_fft_abs_ort(x: NDArray[Any, numpy.float32],
                       ) -> NDArray[Any, numpy.float32]:
    "onnx fft + abs"
    return _custom_fft_abs(x)


custom_fft_abs(x)


# In[28]:


get_ipython().run_line_magic('timeit', 'custom_fft_abs_ort(x)')


# *onnxruntime* is faster than numpy in this case.

# In[29]:


get_ipython().run_line_magic('timeit', 'custom_fft_abs_ort(bigx)')


# ### Inside a FunctionTransformer
# 
# The conversion to ONNX fails if the python function is used.

# In[30]:


from mlprodict.onnx_conv import to_onnx

tr = FunctionTransformer(custom_fft_abs_py)
tr.fit(x)

try:
    onnx_model = to_onnx(tr, x)
except Exception as e:
    print(e)


# Now with the onnx version but before, the converter for FunctionTransformer needs to be overwritten to handle this functionality not available in [sklearn-onnx](https://github.com/onnx/sklearn-onnx). These version are automatically called in function [to_onnx](http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnx_conv/convert.html#mlprodict.onnx_conv.convert.to_onnx) from *mlprodict*.

# In[31]:


tr = FunctionTransformer(custom_fft_abs)
tr.fit(x)

onnx_model = to_onnx(tr, x)


# In[32]:


from mlprodict.onnxrt import OnnxInference

oinf = OnnxInference(onnx_model)
y_onx = oinf.run({'X': x})
y_onx['variable']


# In[33]: