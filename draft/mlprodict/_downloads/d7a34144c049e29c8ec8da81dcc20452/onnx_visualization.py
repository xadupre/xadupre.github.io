#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # ONNX visualization
# 
# [ONNX](https://onnx.ai/) is a serialization format for machine learned model. It is a list of mathematical functions used to describe every prediction function for standard and deep machine learning. Module [onnx](https://github.com/onnx/onnx) offers some tools to [display ONNX graph](http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/auto_examples/plot_pipeline.html). [Netron](https://github.com/lutzroeder/netron) is another approach. The following notebooks explore a ligher visualization.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ## Train a model

# In[2]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression(solver='liblinear')
clr.fit(X_train, y_train)


# ## Convert a model

# In[3]:


import numpy
from mlprodict.onnx_conv import to_onnx
model_onnx = to_onnx(clr, X_train.astype(numpy.float32))


# ## Explore it with OnnxInference

# In[4]:


from mlprodict.onnxrt import OnnxInference

sess = OnnxInference(model_onnx)
sess


# In[5]:


print(sess)


# ## dot

# In[6]:


dot = sess.to_dot()
print(dot)


# In[7]:


from jyquickhelper import RenderJsDot
RenderJsDot(dot)  # add local=True if nothing shows up


# ## magic commands
# 
# The module implements a magic command to easily display graphs.

# In[8]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[9]:


# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'model_onnx')


# ## Shape information
# 
# It is possible to use the python runtime to get an estimation of each node shape.

# In[10]:


get_ipython().run_line_magic('onnxview', 'model_onnx -a 1')


# The shape ``(n, 2)`` means a matrix with an indefinite number of rows and 2 columns.

# ## runtime
# 
# Let's compute the prediction using a Python runtime.

# In[11]:


prob = sess.run({'X': X_test})['output_probability']
prob[:5]


# In[12]:


import pandas
prob = pandas.DataFrame(list(prob)).values
prob[:5]


# Which we compare to the original model.

# In[13]:


clr.predict_proba(X_test)[:5]


# Some time measurement...

# In[14]:


get_ipython().run_line_magic('timeit', 'clr.predict_proba(X_test)')


# In[15]:


get_ipython().run_line_magic('timeit', "sess.run({'X': X_test})['output_probability']")


# With one observation:

# In[16]:


get_ipython().run_line_magic('timeit', 'clr.predict_proba(X_test[:1])')


# In[17]:


get_ipython().run_line_magic('timeit', "sess.run({'X': X_test[:1]})['output_probability']")


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


from pyquickhelper.pycode.profiling import profile
pr, df = profile(lambda: sess.run({'X': X_test})['output_probability'], as_df=True)
ax = df[['namefct', 'cum_tall']].head(n=20).set_index('namefct').plot(kind='bar', figsize=(12, 3), rot=30)
ax.set_title("example of a graph")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right');


# ## Add metadata
# 
# It is possible to add metadata once the model is converted.

# In[20]:


meta = model_onnx.metadata_props.add()
meta.key = "key_meta"
meta.value = "value_meta"


# In[21]:


list(model_onnx.metadata_props)


# In[22]:


model_onnx.metadata_props[0]


# ## Simple PCA

# In[23]:


from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X)


# In[24]:


pca_onnx = to_onnx(model, X.astype(numpy.float32))


# In[25]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[26]:


get_ipython().run_line_magic('onnxview', 'pca_onnx -a 1')


# The graph would probably be faster if the multiplication was done before the subtraction because it is easier to do this one inline than the multiplication.

# In[27]: