#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Converts a logistic regression into C
# 
# The logistic regression is trained in python and executed in C.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ## Train a linear regression

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
y[y == 2] = 1
lr = LogisticRegression()
lr.fit(X, y)


# ## Export  into C

# In[3]:


# grammar is the expected scoring model.
from mlprodict.grammar_sklearn import sklearn2graph
gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])
gr


# We can even check what the function should produce as a score. Types are strict.

# In[4]:


import numpy
X = numpy.array([[numpy.float32(1), numpy.float32(2)]])
e2 = gr.execute(Features=X[0, :])
print(e2)


# We compare with scikit-learn.

# In[5]:


lr.decision_function(X[0:1, :])


# Conversion into C:

# In[6]:


res = gr.export(lang='c', hook={'array': lambda v: v.tolist(), 'float32': lambda v: float(v)})
print(res["code"])


# We execute the code with module [cffi](https://cffi.readthedocs.io/en/latest/).

# In[7]:


from mlprodict.grammar_sklearn.cc import compile_c_function
fct = compile_c_function(res["code"], 2)
fct


# In[8]:


e2 = fct(X[0, :])
e2


# ## Time comparison

# In[9]:


get_ipython().run_line_magic('timeit', 'lr.decision_function(X[0:1, :])')


# In[10]:


get_ipython().run_line_magic('timeit', 'fct(X[0, :])')


# There is a significant speedup on this example. It could be even faster by removing some Python part and optimizing the code produced by [cffi](https://cffi.readthedocs.io/en/latest/). We can also save the creation of the array which contains the output by reusing an existing one.

# In[11]:


out = fct(X[0, :])


# In[12]:


get_ipython().run_line_magic('timeit', 'fct(X[0, :], out)')


# In[13]: