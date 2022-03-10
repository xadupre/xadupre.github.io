#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Lightgbm, double, discrepencies
# 
# Discrepencies usually happens with [lightgbm](https://lightgbm.readthedocs.io/en/latest/) because its code is used double to represent the threshold of trees as ONNX is using float only. There is no way to fix this discrepencies unless the ONNX implementation of trees is using double.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## Simple regression problem
# 
# Target *y* is multiplied by 10 to increase the absolute discrepencies. Relative discrepencies should not change much.

# In[3]:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(2000, n_features=10)
y *= 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# In[4]:


min(y), max(y)


# ## Training a model
# 
# Let's train many models to see how they behave.

# In[5]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# In[6]:


models = [
    RandomForestRegressor(n_estimators=50, max_depth=8),
    GradientBoostingRegressor(n_estimators=50, max_depth=8),
    HistGradientBoostingRegressor(max_iter=50, max_depth=8),
    LGBMRegressor(n_estimators=50, max_depth=8),
    XGBRegressor(n_estimators=50, max_depth=8),
]


# In[7]:


from tqdm import tqdm
for model in tqdm(models):
    model.fit(X_train, y_train)


# ## Conversion to ONNX
# 
# We use function *to_onnx* from this package to avoid the trouble of registering converters from *onnxmltools* for *lightgbm* and *xgboost* libraries.

# In[8]:


from mlprodict.onnx_conv import to_onnx
import numpy
onnx_models = [to_onnx(model, X_train[:1].astype(numpy.float32), rewrite_ops=True)
               for model in models]


# In[9]:


simple_onx = to_onnx(LGBMRegressor(n_estimators=3, max_depth=4).fit(X_train, y_train),
                     X_train[:1].astype(numpy.float32), rewrite_ops=True)
get_ipython().run_line_magic('onnxview', 'simple_onx')


# ## Discrepencies with float32

# In[10]:


from onnxruntime import InferenceSession
from pandas import DataFrame


def max_discrepency(X, skl_model, onx_model):
    expected = skl_model.predict(X).ravel()
    
    sess = InferenceSession(onx_model.SerializeToString())
    got = sess.run(None, {'X': X})[0].ravel()
    
    diff = numpy.abs(got - expected).max()
    return diff


obs = []
x32 = X_test.astype(numpy.float32)
for model, onx in zip(models, onnx_models):
    diff = max_discrepency(x32, model, onx)
    obs.append(dict(name=model.__class__.__name__, max_diff=diff))

    
DataFrame(obs)


# In[11]:


DataFrame(obs).set_index("name").plot(kind="bar").set_title("onnxruntime + float32");


# ## Discrepencies with mlprodict
# 
# This is not available with the current standard ONNX specifications. It required *mlprodict* to implement a runtime for tree ensemble supporting doubles.

# In[12]:


from mlprodict.onnxrt import OnnxInference
from pandas import DataFrame


def max_discrepency_2(X, skl_model, onx_model):
    expected = skl_model.predict(X).ravel()
    
    sess = OnnxInference(onx_model)
    got = sess.run({'X': X})['variable'].ravel()
    
    diff = numpy.abs(got - expected).max()
    return diff


obs = []
x32 = X_test.astype(numpy.float32)
for model, onx in zip(models, onnx_models):
    diff = max_discrepency_2(x32, model, onx)
    obs.append(dict(name=model.__class__.__name__, max_diff=diff))

    
DataFrame(obs)


# In[13]:


DataFrame(obs).set_index("name").plot(kind="bar").set_title("mlprodict + float32");


# ## Discrepencies with mlprodict and double
# 
# The conversion needs to happen again.

# In[14]:


simple_onx = to_onnx(LGBMRegressor(n_estimators=2, max_depth=2).fit(X_train, y_train),
                     X_train[:1].astype(numpy.float64), rewrite_ops=True)
get_ipython().run_line_magic('onnxview', 'simple_onx')


# In[15]:


onnx_models_64 = []
for model in tqdm(models):
    onx = to_onnx(model, X_train[:1].astype(numpy.float64), rewrite_ops=True)
    onnx_models_64.append(onx)


# In[16]:


obs64 = []
x64 = X_test.astype(numpy.float64)
for model, onx in zip(models, onnx_models_64):
    oinf = OnnxInference(onx)
    diff = max_discrepency_2(x64, model, onx)
    obs64.append(dict(name=model.__class__.__name__, max_diff=diff))

    
DataFrame(obs64)


# In[17]:


DataFrame(obs64).set_index("name").plot(kind="bar").set_title("mlprodict + float64");


# In[18]:


df = DataFrame(obs).set_index('name').merge(DataFrame(obs64).set_index('name'),
                                              left_index=True, right_index=True)
df.columns = ['float32', 'float64']
df


# In[19]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df.plot(kind="bar", ax=ax[0]).set_title("mlprodict")
df.plot(kind="bar", ax=ax[1], logy=True).set_title("mlprodict");


# The runtime using double produces lower discrepencies except for *xgboost*. It is probably using float and all the others are using double.
# 
# **Note:** function [to_onnx](http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnx_conv/convert.html#mlprodict.onnx_conv.convert.to_onnx) automatically registers converters for *lightgbm*, *xgboost* and a dedicated runtime for a new ONNX node [TreeEnsembleRegressorDouble](http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnxrt/ops_cpu/op_tree_ensemble_regressor.html#mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor.TreeEnsembleRegressorDouble). It uses [skl2onnx.to_onnx](https://onnx.ai/sklearn-onnx/api_summary.html#skl2onnx.to_onnx) underneath.

# In[20]: