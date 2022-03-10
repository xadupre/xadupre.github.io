#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Profiling with onnxruntime
# 
# The notebook profiles the execution of an ONNX graph built from a *KMeans* model and executed with *onnxruntime*. It then study the decomposition of one einsum equation into more simple operators.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## KMeans

# ### Builds a KMeans

# In[4]:


from sklearn.datasets import make_classification
X, y = make_classification(100000)


# In[5]:


from sklearn.cluster import KMeans
km = KMeans(max_iter=10)
km.fit(X)


# In[6]:


import numpy
from mlprodict.onnx_conv import to_onnx
onx = to_onnx(km, X[:1].astype(numpy.float32))


# In[7]:


get_ipython().run_line_magic('onnxview', 'onx')


# ### Json
# 
# Another way to look into a model.

# In[8]:


from mlprodict.onnxrt import OnnxInference

oinf = OnnxInference(onx)
js = oinf.to_json()


# In[9]:


import json
from io import StringIO
from jyquickhelper import JSONJS
JSONJS(json.load(StringIO(oinf.to_json())))


# ### Profiling

# In[10]:


from mlprodict.onnxrt import OnnxInference

oinf = OnnxInference(onx, runtime="onnxruntime1",
                     runtime_options={"enable_profiling": True})


# In[11]:


for i in range(0, 111):
    oinf.run({"X": X.astype(numpy.float32)})


# In[12]:


df = oinf.get_profiling(as_df=True)
df


# In[13]:


import matplotlib.pyplot as plt
gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_n.plot.barh(ax=ax[1])
ax[0].set_title("duration")
ax[1].set_title("n occurences");


# In[14]:


gr2 = df.loc[(df.args_op_name == 'Add') & (df.dur > 10), ['dur', "name"]].groupby("name").sum().sort_values('dur')
gr2.plot.barh(figsize=(10, 4));


# ### onnxruntime

# In[15]:


from onnxruntime import InferenceSession, RunOptions, SessionOptions
so = SessionOptions()
so.enable_profiling = True
sess = InferenceSession(onx.SerializeToString(), so)


# In[16]:


for i in range(0, 111):
    sess.run(None, {'X': X.astype(numpy.float32)}, )


# In[17]:


prof = sess.end_profiling()
prof


# In[18]:


with open(prof, "r") as f:
    js = json.load(f)
    
js[:3]


# In[19]:


from pandas import DataFrame
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession

df = DataFrame(OnnxWholeSession.process_profiling(js))
df


# ## Einsum: `bsnh,btnh->bnts`
# 
# This section looks into the ONNX graph produces by the decomposition of an einsum equation into more simple ONNX operator (no einsum).

# ### Three implementations

# In[20]:


from mlprodict.testing.einsum import einsum as onx_einsum
from mlprodict.testing.einsum.einsum_fct import _einsum, enumerate_cached_einsum
from numpy import einsum as np_einsum


# First classic numpy.

# In[21]:


equation = "bsnh,btnh->bnts"

N = 2
inputs = [numpy.random.randn(N, N, N, N).astype(numpy.float32),
          numpy.random.randn(N, N, N, N).astype(numpy.float32)]
np_einsum(equation, *inputs)


# Then einsum executed by *onnxruntime*:

# In[22]:


onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=True, verbose=1, decompose=False)


# In[23]:


obj = _einsum(equation, runtime='onnxruntime1', optimize=True, verbose=1,
              decompose=False, dtype=inputs[0].dtype)


# In[24]:


get_ipython().run_line_magic('onnxview', 'obj.onnx_')


# Same equation but decomposed.

# In[25]:


obj = _einsum(equation, runtime='onnxruntime1', optimize=True, verbose=1,
              decompose=True, dtype=inputs[0].dtype)


# In[26]:


get_ipython().run_line_magic('onnxview', 'obj.onnx_')


# In[27]:


onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=True, verbose=1)


# ### First benchmark

# In[28]:


N = 20
inputs = [numpy.random.randn(N, N, N, N).astype(numpy.float32),
          numpy.random.randn(N, N, N, N).astype(numpy.float32)]


# *numpy.einsum*

# In[29]:


get_ipython().run_line_magic('timeit', 'numpy.einsum(equation, *inputs)')


# *onnxruntime einsum*

# In[30]:


get_ipython().run_line_magic('timeit', "onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=True, verbose=1, decompose=False)")


# *onnxruntime decomposed einsum*

# In[31]:


get_ipython().run_line_magic('timeit', "onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=True, verbose=1)")


# Let's disable the optimization to see the difference. The optimization goes through all the permutation of the letters of the equation and compares the computation time to find the best one.

# In[32]:


get_ipython().run_line_magic('timeit', "onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=False, verbose=1, decompose=False)")


# It has no significant impact here but it has for the decomposition. The not optimized version is much slower.

# In[33]:


get_ipython().run_line_magic('timeit', "onx_einsum(equation, *inputs, runtime='onnxruntime1', optimize=False, verbose=1)")


# ### Profiling of the not optimized version
# 
# Let's profile the graph obtained with the decomposition.

# In[34]:


obj = _einsum(equation, runtime='onnxruntime1', optimize=False, verbose=1,
              decompose=True, dtype=inputs[0].dtype)
onx = obj.onnx_


# In[35]:


obj.equation, obj.equation_


# In[36]:


from mlprodict.onnxrt import OnnxInference

oinf = OnnxInference(onx, runtime="onnxruntime1",
                     runtime_options={"enable_profiling": True})

d_inputs = {'X0': inputs[0], 'X1': inputs[1]}
for i in range(0, 100):
    oinf.run(d_inputs)
    
df = oinf.get_profiling(as_df=True)
df.head()


# In[37]:


import matplotlib.pyplot as plt
gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_n.plot.barh(ax=ax[1])
ax[0].set_title("duration - not optimized - %s" % obj.equation_)
ax[1].set_title("n occurences");


# ### Profiling of the optimized version

# In[38]:


obj = _einsum(equation, runtime='onnxruntime1', optimize=True, verbose=1,
              decompose=True, dtype=inputs[0].dtype)
onx = obj.onnx_


# In[39]:


obj.equation, obj.equation_


# The second equation is the optimized equation.

# In[40]:


from mlprodict.onnxrt import OnnxInference

oinf = OnnxInference(onx, runtime="onnxruntime1",
                     runtime_options={"enable_profiling": True})

d_inputs = {'X0': inputs[0], 'X1': inputs[1]}
for i in range(0, 100):
    oinf.run(d_inputs)
    
df = oinf.get_profiling(as_df=True)
df.head()


# In[41]:


gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_n.plot.barh(ax=ax[1])
ax[0].set_title("duration - optimized - %s" % obj.equation_)
ax[1].set_title("n occurences");


# onnxruntime was able to fuse MatMul with a transposition. That explains why it is faster.

# In[42]:


gr_dur = df[['dur', "args_op_name", "name"]].groupby(["args_op_name", "name"], as_index=False).sum().sort_values('dur')
gr_dur


# In[43]:


gr_dur[gr_dur.args_op_name == "Transpose"]


# Let's draw again the graph to see which transpose is is which.

# In[44]:


get_ipython().run_line_magic('onnxview', 'onx')


# The optimized looked into all permutations. We see that the letter ordering should be carefully chosen.

# In[45]:


import pandas
df = pandas.DataFrame(obj.timed_permutations_, columns=["time", "equation"])
df = df.sort_values('time')
df = df.set_index("equation")
ax = df.plot.barh(figsize=(8, 25))
ax.set_title("%s OPTIMIZED INTO %s" % (obj.equation, obj.equation_));


# In[46]: