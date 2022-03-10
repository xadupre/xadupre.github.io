#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Memory usage
# 
# The [first benchmark](http://www.xavierdupre.fr/app/_benchmarks/helpsphinx/sklbench_results/index.html) based on [scikti-learn's benchmark](https://github.com/jeremiedbb/scikit-learn_benchmarks) shows high peaks of memory usage for the python runtime on linear models. Let's see how to measure that.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ## Artificial huge data 

# In[2]:


import numpy
N, nfeat = 300000, 200
N * nfeat * 8 / 1e9


# In[3]:


X = numpy.random.random((N, nfeat))
y = numpy.empty((N, 50))
for i in range(y.shape[1]):
    y[:, i] = X.sum(axis=1) + numpy.random.random(N)
X.shape, y.shape


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)


# In[5]:


from sklearn.linear_model import LinearRegression
clr = LinearRegression()
clr.fit(X_train, y_train)


# In[6]:


from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
clr_onnx = to_onnx(clr, X_train[:1].astype(numpy.float32))
oinfpy = OnnxInference(clr_onnx, runtime='python')


# Let's minimize the cost of verifications on scikit-learn's side.

# In[7]:


from sklearn import set_config
set_config(assume_finite=True)


# ## Profiling the prediction function

# In[8]:


from pyquickhelper.pycode.profiling import profile
print(profile(lambda: clr.predict(X_test), 
              pyinst_format='text')[1])


# In[9]:


import numpy

def nastype32(mat):
    return mat.astype(numpy.float32)

print(profile(lambda: oinfpy.run({'X': nastype32(X_test)}), 
              pyinst_format='text')[1])


# Most of the time is taken out into casting into float. Let's take it out.

# In[10]:


X_test32 = X_test.astype(numpy.float32)

print(profile(lambda: oinfpy.run({'X': X_test32}), 
              pyinst_format='text')[1])


# Much better.

# ## SGDClasifier
# 
# This models is implemented with many ONNX nodes. Let's how it behaves.

# In[11]:


from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
data = load_iris()
Xir, yir = data.data, data.target
Xir_train, Xir_test, yir_train, yir_test = train_test_split(Xir, yir)
sgcl = SGDClassifier()
sgcl.fit(Xir_train, yir_train)


# In[12]:


sgd_onnx = to_onnx(sgcl, Xir_train.astype(numpy.float32))


# In[13]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[14]:


get_ipython().run_line_magic('onnxview', 'sgd_onnx')


# In[15]:


sgd_oinf = OnnxInference(sgd_onnx)


# In[16]:


def call_n_times_x1(n, X_test, sgd_oinf):
    for i in range(n):
        res = sgd_oinf.run({'X': X_test})
    return res

call_n_times_x1(20, Xir_test[:1].astype(numpy.float32), sgd_oinf)


# In[17]:


sgcl.decision_function(Xir_test[:1])


# In[18]:


xir_32 = Xir_test[:1].astype(numpy.float32)

print(profile(lambda: call_n_times_x1(20000, xir_32, sgd_oinf), 
              pyinst_format='text')[1])


# The code in ``mlprodict/onnxrt/onnx_inference_node.py`` just calls an operator and updates the list containing all the results. The time in here is significant if the number of node is huge if the python runtime is used.

# ## Memory profiling

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


from memory_profiler import memory_usage
memprof_skl = memory_usage((clr.predict, (X_test, )), timestamps=True, interval=0.01)


# In[21]:


memprof_skl


# In[22]:


import matplotlib.pyplot as plt
from pandas import DataFrame, to_datetime

def mem_profile_plot(mem, title):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    df = DataFrame(mem, columns=["memory", "timestamp"])
    df["timestamp"] = to_datetime(df.timestamp)
    df["timestamp"] -= df.timestamp.min()
    df.set_index("timestamp").plot(ax=ax)
    ax.set_title(title + "\nmemory usage")
    return ax

mem_profile_plot(memprof_skl, "clr.predict");


# In[23]:


memprof_onx = memory_usage((oinfpy.run, ({'X': X_test32}, )), timestamps=True, interval=0.01)
mem_profile_plot(memprof_onx, "oinfpy.run");


# In[24]:


memprof_onx2 = memory_usage((oinfpy.run, ({'X': X_test.astype(numpy.float32, copy=False)}, )),
                           timestamps=True, interval=0.01)
mem_profile_plot(memprof_onx2, "oinfpy.run + astype(numpy.float32)");


# This is not very informative.

# ## Memory profiling outside the notebook
# 
# More precise.

# In[25]:


get_ipython().run_cell_magic('writefile', 'mprof_clr_predict.py', '\nimport numpy\nN, nfeat = 300000, 200\nX = numpy.random.random((N, nfeat))\ny = numpy.empty((N, 50))\nfor i in range(y.shape[1]):\n    y[:, i] = X.sum(axis=1) + numpy.random.random(N)\n    \nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)   \n\nfrom sklearn.linear_model import LinearRegression\nclr = LinearRegression()\nclr.fit(X_train, y_train)\n\nfrom sklearn import set_config\nset_config(assume_finite=True)    \n\nfrom memory_profiler import profile\n@profile\ndef clr_predict():\n    clr.predict(X_test)\n    \nclr_predict()')


# In[26]:


get_ipython().system('python -m memory_profiler mprof_clr_predict.py --timestamp')


# The notebook seems to increase the memory usage.

# In[27]:


get_ipython().run_cell_magic('writefile', 'mprof_onnx_run.py', "\nimport numpy\nN, nfeat = 300000, 200\nX = numpy.random.random((N, nfeat))\ny = numpy.empty((N, 50))\nfor i in range(y.shape[1]):\n    y[:, i] = X.sum(axis=1) + numpy.random.random(N)\n    \nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)   \n\nfrom sklearn.linear_model import LinearRegression\nclr = LinearRegression()\nclr.fit(X_train, y_train)\n\nfrom mlprodict.onnx_conv import to_onnx\nfrom mlprodict.onnxrt import OnnxInference\nclr_onnx = to_onnx(clr, X_train[:1].astype(numpy.float32))\noinfpy = OnnxInference(clr_onnx, runtime='python')\nX_test32 = X_test.astype(numpy.float32)\n\nfrom sklearn import set_config\nset_config(assume_finite=True)    \n\nfrom memory_profiler import profile\n@profile\ndef oinfpy_predict():\n    oinfpy.run({'X': X_test32})\n    \noinfpy_predict()")


# In[28]:


get_ipython().system('python -m memory_profiler mprof_onnx_run.py --timestamp')


# In[29]:


get_ipython().run_cell_magic('writefile', 'mprof_onnx_run32.py', "\nimport numpy\nN, nfeat = 300000, 200\nX = numpy.random.random((N, nfeat))\ny = numpy.empty((N, 50))\nfor i in range(y.shape[1]):\n    y[:, i] = X.sum(axis=1) + numpy.random.random(N)\n    \nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)   \n\nfrom sklearn.linear_model import LinearRegression\nclr = LinearRegression()\nclr.fit(X_train, y_train)\n\nfrom mlprodict.onnx_conv import to_onnx\nfrom mlprodict.onnxrt import OnnxInference\nclr_onnx = to_onnx(clr, X_train[:1].astype(numpy.float32))\noinfpy = OnnxInference(clr_onnx, runtime='python')\n\nfrom sklearn import set_config\nset_config(assume_finite=True)    \n\nfrom memory_profiler import profile\n@profile\ndef oinfpy_predict32():\n    oinfpy.run({'X': X_test.astype(numpy.float32)})\n    \noinfpy_predict32()")


# In[30]:


get_ipython().system('python -m memory_profiler mprof_onnx_run32.py --timestamp')


# In[31]: