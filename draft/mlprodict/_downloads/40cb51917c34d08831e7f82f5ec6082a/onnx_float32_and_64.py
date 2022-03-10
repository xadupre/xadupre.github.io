#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # ONNX graph, single or double floats
# 
# The notebook shows discrepencies obtained by using double floats instead of single float in two cases. The second one involves [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ## Simple case of a linear regression
# 
# A linear regression is simply a matrix multiplication followed by an addition: $Y=AX+B$. Let's train one with [scikit-learn](https://scikit-learn.org/stable/).

# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LinearRegression()
clr.fit(X_train, y_train)


# In[3]:


clr.score(X_test, y_test)


# In[4]:


clr.coef_


# In[5]:


clr.intercept_


# Let's predict with *scikit-learn* and *python*.

# In[6]:


ypred = clr.predict(X_test)
ypred[:5]


# In[7]:


py_pred = X_test @ clr.coef_ + clr.intercept_
py_pred[:5]


# In[8]:


clr.coef_.dtype, clr.intercept_.dtype


# ## With ONNX
# 
# With *ONNX*, we would write this operation as follows... We still need to convert everything into single floats = float32.

# In[9]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[10]:


from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxAdd
import numpy

onnx_fct = OnnxAdd(OnnxMatMul('X', clr.coef_.astype(numpy.float32), op_version=12),
                   numpy.array([clr.intercept_], dtype=numpy.float32),
                   output_names=['Y'], op_version=12)
onnx_model32 = onnx_fct.to_onnx({'X': X_test.astype(numpy.float32)})

# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'onnx_model32')


# The next line uses a python runtime to compute the prediction.

# In[11]:


from mlprodict.onnxrt import OnnxInference
oinf = OnnxInference(onnx_model32, inplace=False)
ort_pred = oinf.run({'X': X_test.astype(numpy.float32)})['Y']
ort_pred[:5]


# And here is the same with [onnxruntime](https://github.com/microsoft/onnxruntime)...

# In[12]:


from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx
# line needed when onnx is more recent than onnxruntime
onnx_model32.ir_version = get_ir_version_from_onnx()
oinf = OnnxInference(onnx_model32, runtime="onnxruntime1")
ort_pred = oinf.run({'X': X_test.astype(numpy.float32)})['Y']
ort_pred[:5]


# ## With double instead of single float
# 
# [ONNX](https://onnx.ai/) was originally designed for deep learning which usually uses floats but it does not mean cannot be used. Every number is converted into double floats.

# In[13]:


onnx_fct = OnnxAdd(OnnxMatMul('X', clr.coef_.astype(numpy.float64), op_version=12),
                   numpy.array([clr.intercept_], dtype=numpy.float64),
                   output_names=['Y'], op_version=12)
onnx_model64 = onnx_fct.to_onnx({'X': X_test.astype(numpy.float64)})


# And now the *python* runtime...

# In[14]:


oinf = OnnxInference(onnx_model64)
ort_pred = oinf.run({'X': X_test})['Y']
ort_pred[:5]


# And the *onnxruntime* version of it.

# In[15]:


oinf = OnnxInference(onnx_model64, runtime="onnxruntime1")
ort_pred = oinf.run({'X': X_test.astype(numpy.float64)})['Y']
ort_pred[:5]


# ## And now the GaussianProcessRegressor
# 
# This shows a case

# In[16]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
gau = GaussianProcessRegressor(alpha=10, kernel=DotProduct())
gau.fit(X_train, y_train)


# In[17]:


from mlprodict.onnx_conv import to_onnx
onnxgau32 = to_onnx(gau, X_train.astype(numpy.float32))
oinf32 = OnnxInference(onnxgau32, runtime="python", inplace=False)
ort_pred32 = oinf32.run({'X': X_test.astype(numpy.float32)})['GPmean']
numpy.squeeze(ort_pred32)[:25]


# In[18]:


onnxgau64 = to_onnx(gau, X_train.astype(numpy.float64))
oinf64 = OnnxInference(onnxgau64, runtime="python", inplace=False)
ort_pred64 = oinf64.run({'X': X_test.astype(numpy.float64)})['GPmean']
numpy.squeeze(ort_pred64)[:25]


# The differences between the predictions for single floats and double floats...

# In[19]:


numpy.sort(numpy.sort(numpy.squeeze(ort_pred32 - ort_pred64)))[-5:]


# Who's right or wrong... The differences between the predictions with the original model...

# In[20]:


pred = gau.predict(X_test.astype(numpy.float64))


# In[21]:


numpy.sort(numpy.sort(numpy.squeeze(ort_pred32 - pred)))[-5:]


# In[22]:


numpy.sort(numpy.sort(numpy.squeeze(ort_pred64 - pred)))[-5:]


# Double predictions clearly wins.

# In[23]:


# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'onnxgau64')


# ## Saves...
# 
# Let's keep track of it.

# In[24]:


with open("gpr_dot_product_boston_32.onnx", "wb") as f:
    f.write(onnxgau32.SerializePartialToString())
from IPython.display import FileLink
FileLink('gpr_dot_product_boston_32.onnx')


# In[25]:


with open("gpr_dot_product_boston_64.onnx", "wb") as f:
    f.write(onnxgau64.SerializePartialToString())
FileLink('gpr_dot_product_boston_64.onnx')


# ## Side by side
# 
# We may wonder where the discrepencies start. But for that, we need to do a side by side.

# In[26]:


from mlprodict.onnxrt.validate.side_by_side import side_by_side_by_values
sbs = side_by_side_by_values([(oinf32, {'X': X_test.astype(numpy.float32)}),
                              (oinf64, {'X': X_test.astype(numpy.float64)})])

from pandas import DataFrame
df = DataFrame(sbs)
# dfd = df.drop(['value[0]', 'value[1]', 'value[2]'], axis=1).copy()
df


# The differences really starts for output ``'O0'`` after the matrix multiplication. This matrix melts different number with very different order of magnitudes and that alone explains the discrepencies with doubles and floats on that particular model.

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
ax = df[['name', 'v[1]']].iloc[1:].set_index('name').plot(kind='bar', figsize=(14,4), logy=True)
ax.set_title("Relative differences for each output between float32 and "
             "float64\nfor a GaussianProcessRegressor");


# Before going further, let's check how sensitive the trained model is about converting double into floats.

# In[28]:


pg1 = gau.predict(X_test)
pg2 = gau.predict(X_test.astype(numpy.float32).astype(numpy.float64))
numpy.sort(numpy.sort(numpy.squeeze(pg1 - pg2)))[-5:]


# Having float or double inputs should not matter. We confirm that with the model converted into ONNX.

# In[29]:


p1 = oinf64.run({'X': X_test})['GPmean']
p2 = oinf64.run({'X': X_test.astype(numpy.float32).astype(numpy.float64)})['GPmean']
numpy.sort(numpy.sort(numpy.squeeze(p1 - p2)))[-5:]


# Last verification.

# In[30]:


sbs = side_by_side_by_values([(oinf64, {'X': X_test.astype(numpy.float32).astype(numpy.float64)}),
                              (oinf64, {'X': X_test.astype(numpy.float64)})])
df = DataFrame(sbs)
ax = df[['name', 'v[1]']].iloc[1:].set_index('name').plot(kind='bar', figsize=(14,4), logy=True)
ax.set_title("Relative differences for each output between float64 and float64 rounded to float32"
             "\nfor a GaussianProcessRegressor");


# In[31]: