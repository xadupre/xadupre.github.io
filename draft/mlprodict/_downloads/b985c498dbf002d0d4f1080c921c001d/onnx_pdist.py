#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Pairwise distances with ONNX (pdist)
# 
# Function [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) computes pairwise distances between observations in n-dimensional space. It is not that difficult to convert that into *ONNX* when the dimension of the input is always the same. What if not?

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## Function pdist
# 
# The function [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) distances. Let's denote a list of vectors $(X_1, ..., X_n)$, function [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) returns the matrix $D=(d_{ij})$ where $d_{ij}=dist(X_i, X_j)=\lVert X_i - X_j \rVert^2$.

# In[3]:


import numpy
from scipy.spatial.distance import pdist, squareform

M = numpy.array([[0, 1],
                 [1, 2],
                 [0.1, 1.1],
                 [2, 2]], dtype=float)

d1 = squareform(pdist(M, metric='sqeuclidean'))
d1


# The two following functions are implemented to reduce the number of allocations the algorithm requires.

# In[4]:


def custom_pdist(M):
    n = M.shape[0]
    res = numpy.zeros((n, n))
    buffer = numpy.empty(M.shape)
    for i in range(n):
        numpy.subtract(M, M[i], out=buffer)  # broadcasted substraction
        numpy.square(buffer, out=buffer)
        res[i, :] = numpy.sum(buffer, axis=1)
    return res

d2 = custom_pdist(M)
d2


# This function computes $n^2$ distances wheres only $\frac{n(n-1)}{2}$ are necessary since the final matrix is symmetric. Let's change the implementation to reflect that.

# In[5]:


def custom_pdist_lower(M):
    n = M.shape[0]
    res = numpy.zeros((n, n))
    buffer = numpy.empty((M.shape[0]-1, M.shape[1]))
    a = numpy.empty(M.shape[0])
    for i in range(1, n):
        numpy.subtract(M[:i], M[i], out=buffer[:i])  # broadcasted substraction
        numpy.square(buffer[:i], out=buffer[:i])
        numpy.sum(buffer[:i], axis=1, out=a[:i])
        res[:i, i] = a[:i]
        res[i, :i] = a[:i]
    return res

d3 = custom_pdist_lower(M)
d3


# ## Loop mechanism in ONNX
# 
# Operator [Loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop) seems appropriate but it is just a loop wheras [Scan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scan) holds accumulator. The first graph is what is repeated inside the loop.

# In[6]:


from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxIdentity, OnnxScan
from skl2onnx.common.data_types import FloatTensorType

initial = numpy.array([0, 0]).astype(numpy.float32).reshape((2,))
x = numpy.array([1, 2, 3, 4, 5, 6]).astype(numpy.float32).reshape((3, 2))

add_node = OnnxAdd('sum_in', 'next', output_names=['sum_out'], op_version=12)
id_node = OnnxIdentity(add_node, output_names=['scan_out'], op_version=12)

scan_body = id_node.to_onnx(
    {'sum_in': initial, 'next': initial},
    outputs=[('sum_out', FloatTensorType()),
             ('scan_out', FloatTensorType())])

# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'scan_body')


# The operator [Scan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scan) repeats this graph a couple of times. *sum_in* is an accumulator, *next* is the iterated row from the input matrix.

# In[7]:


node = OnnxScan('initial', 'x', output_names=['y', 'z'],
                num_scan_inputs=1, body=scan_body.graph)

model_def = node.to_onnx(
    {'initial': initial, 'x': x},
    outputs=[('y', FloatTensorType()),
             ('z', FloatTensorType())])

# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'model_def')


# All together in the same graph.

# In[8]:


# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'model_def -r 1')


# In[9]:


from mlprodict.onnxrt import OnnxInference
oinf = OnnxInference(model_def)
res = oinf.run({'initial': initial, 'x': x})
res['y']


# In[10]:


res['z']


# ## Back to pdist
# 
# [sklearn-onnx](https://github.com/onnx/sklearn-onnx) implements function *pdist* with *ONNX* operators. The parameter ``inputs=[('x', FloatTensorType())`` tels the method ``to_onnx`` that the dimension of the inputs is not fixed and should not be checked.

# In[11]:


# from skl2onnx.algebra.complex_functions import squareform_pdist_

from collections import OrderedDict
from skl2onnx.algebra.onnx_ops import (
    OnnxSub, OnnxReduceSumSquare, OnnxSqueeze,
    OnnxIdentity, OnnxScan)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.tools import get_opset_number_from_onnx


def squareform_pdist(X, **kwargs):
    """Returns the ONNX graph which computes
    ``squareform(pdist(X, metric='sqeuclidean')``."""

    # The subgraph executed at every iteration.
    opv = get_opset_number_from_onnx()
    diff = OnnxSub('next_in', 'next', output_names=['diff'], op_version=opv)
    id_next = OnnxIdentity('next_in', output_names=['next_out'], op_version=opv)
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1], op_version=opv)
    flat = OnnxSqueeze(norm, numpy.array([1], dtype=numpy.int64),
                       output_names=['scan_out'], op_version=opv)
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        # Size must be empty otherwise onnxruntime fails
        # at execution time if it receives a matrix
        # with a different shape. With 'None', the same ONNX graph
        # can compute pairwise distance for any shape.
        outputs=[('next_out', FloatTensorType([None, None])),
                 ('scan_out', FloatTensorType([None]))],
        other_outputs=[flat])

    # The loop.
    # 'scan0_{idself}' means the variable name will include
    # id(OnnxScan), this is needed if squareform_pdist is used
    # twice in the same graph.
    node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph, op_version=opv,
                    **kwargs)
    return node[1]    

opv = get_opset_number_from_onnx()
onnx_fct = OnnxIdentity(squareform_pdist('x'), output_names='Y', op_version=opv)
model_def = onnx_fct.to_onnx(inputs=[('x', FloatTensorType())])

# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'model_def')


# In[12]:


from collections import OrderedDict
from skl2onnx.algebra.onnx_ops import (
    OnnxSub, OnnxReduceSumSquare, OnnxSqueeze,
    OnnxIdentity, OnnxScan)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.tools import get_opset_number_from_onnx


def squareform_pdist(X, **kwargs):
    # The subgraph executed at every iteration.
    opv = get_opset_number_from_onnx()
    diff = OnnxSub('next_in', 'next', output_names=['diff'], op_version=opv)
    id_next = OnnxIdentity('next_in', output_names=['next_out'], op_version=opv)
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1], op_version=opv)
    flat = OnnxSqueeze(norm, numpy.array([1], dtype=numpy.int64),
                       output_names=['scan_out'], op_version=opv)
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        outputs=[('next_out', FloatTensorType([None, None])),
                 ('scan_out', FloatTensorType([None]))],
        other_outputs=[flat])

    # The loop.
    node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph, op_version=opv,
                    **kwargs)
    return node[1]    

opv = get_opset_number_from_onnx()
onnx_fct = OnnxIdentity(squareform_pdist('x'), output_names='Y', op_version=opv)
model_def = onnx_fct.to_onnx(inputs=[('x', FloatTensorType())])


# Notice the double arrow. Input _x_ is used twice, once as an permanent state involved in broacasted substract, another time to iterator rows. On the other side, the first output of operator *Scan* is a permanent state equal to the input, the second one is an aggregation of results produced at each iteration. Each of those produces a row of a final matrix.

# In[13]:


oinf = OnnxInference(model_def)
body = oinf['Sc_Scan', 'body']

# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'body.g')


# All together.

# In[14]:


# add -l 1 if nothing shows up
get_ipython().run_line_magic('onnxview', 'model_def -r 1')


# Let's now execute the graph and compare it with the original graph.

# In[15]:


d1 = squareform(pdist(M, metric='sqeuclidean'))
d1


# In[16]:


oinf.run({'x': M})['Y']


# In[17]:


get_ipython().run_line_magic('timeit', "squareform(pdist(M, metric='sqeuclidean'))")


# In[18]:


get_ipython().run_line_magic('timeit', 'custom_pdist(M)')


# In[19]:


get_ipython().run_line_magic('timeit', 'custom_pdist_lower(M)')


# In[20]:


get_ipython().run_line_magic('timeit', "oinf.run({'x': M})['Y']")


# In[21]:


M32 = M.astype(numpy.float32)


# In[22]:


from mlprodict.tools import get_ir_version_from_onnx
model_def.ir_version = get_ir_version_from_onnx()


# In[23]:


oinfrt = OnnxInference(model_def, runtime="onnxruntime1")
oinfrt.run({'x': M32})['Y']


# In[24]:


get_ipython().run_line_magic('timeit', "oinfrt.run({'x': M32})['Y']")


# ## Benchmark

# In[25]:


from timeit import Timer


def measure_time(name, stmt, context, repeat=10, number=10):
    tim = Timer(stmt, globals=context)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res ** 2)
    dev = (dev - mean**2) ** 0.5
    return dict(average=mean, deviation=dev, min_exec=numpy.min(res),
                max_exec=numpy.max(res), repeat=repeat, number=number,
                nrows=context['M'].shape[0], ncols=context['M'].shape[1],
                name=name)

measure_time("scipy", "squareform(pdist(M, metric='sqeuclidean'))",
             context={'squareform': squareform, 'M': M,
                      'pdist': pdist})


# In[26]:


from tqdm import trange

def generator():
    for feat in [5, 10, 50, 100]:
        for n in [5, 10, 20, 50, 100, 400, 1000]:
            if n <= 500 or feat <= 10:
                yield feat, n
            
all_values = list(generator())

rows = []

with trange(len(all_values)) as t:
    for i in t:        
        feat, n = all_values[i]
        t.set_description("feat=%d n=%d" % (feat, n))
        M = numpy.random.rand(n, feat)

        context = {'squareform': squareform, 'M': M, 'pdist': pdist}
        res = measure_time("scipy", "squareform(pdist(M, metric='sqeuclidean'))", context=context)
        res['dimres'] = squareform(pdist(M, metric='sqeuclidean')).shape[0]
        rows.append(res)

        context = {'M': M, 'custom_pdist': custom_pdist}
        res = measure_time("numpy", "custom_pdist(M)", context=context)
        res['dimres'] = custom_pdist(M).shape[0]
        rows.append(res)

        context = {'M': M, 'custom_pdist_lower': custom_pdist_lower}
        res = measure_time("numpy-lower", "custom_pdist_lower(M)", context=context)
        res['dimres'] = custom_pdist_lower(M).shape[0]
        rows.append(res)

        context = {'oinf': oinf, 'M': M}
        res = measure_time("onnx-py", "oinf.run({'x': M})['Y']", context=context)
        res['dimres'] = oinf.run({'x': M})['Y'].shape[0]
        rows.append(res)

        M32 = M.astype(numpy.float32)
        context = {'oinfrt': oinfrt, 'M': M32}
        res = measure_time("onnx-rt", "oinfrt.run({'x': M})['Y']", context=context)
        res['dimres'] = oinfrt.run({'x': M32})['Y'].shape[0]
        rows.append(res)

    
from pandas import DataFrame
df = DataFrame(rows)
df.head()


# In[27]:


from pandas import pivot_table
piv = pivot_table(df, index=["nrows"], columns= ['ncols', 'name'], values='average')
piv.head().T


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for i, ncol in enumerate([10, 50, 100]):
    piv = df[df.ncols==ncol].pivot("nrows", "name", "average")
    piv.plot(ax=ax[i], logy=True, logx=True)
    ax[i].set_title("ncol=%d" % ncol)
ax;


# Curves are not linear and rather difficult to interpret. The algorithm *numpy-lower* and *scipy* should be close as the cost of both algorithm are similar. However, *scipy* reduces the number of trips between C and python. The C implementation of the distance is here:
# [sqeuclidean_distance_double](https://github.com/scipy/scipy/blob/master/scipy/spatial/src/distance_impl.h#L50). The final cost is a combination of computation, multithreading, allocations...

# In[30]:


from pyquickhelper.pycode.profiling import profile
M = numpy.random.rand(100, 10)

pr1, df1 = profile(lambda: [squareform(pdist(M, metric='sqeuclidean')) for i in range(0, 1000)],
                   as_df=True)
pr2, df2 = profile(lambda: [custom_pdist_lower(M) for i in range(0, 1000)], as_df=True)


# In[31]:


ax = df1[['namefct', 'cum_tall']].head(n=15).set_index('namefct').plot(
    kind='bar', figsize=(8, 3), rot=30)
ax.set_title("scipy")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right')


# In[32]:


ax = df2[['namefct', 'cum_tall']].head(n=15).set_index('namefct').plot(
    kind='bar', figsize=(8, 3), rot=30)
ax.set_title("numpy-lower")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right');    


# [Universal function](https://docs.scipy.org/doc/numpy/reference/ufuncs.html) do not seem to be very efficient in our case. The last graph shows time ratio between implementations of *pdist* and the baseline *scipy*.

# In[33]:


fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for i, ncol in enumerate([10, 50, 100]):
    piv = df[df.ncols==ncol].pivot("nrows", "name", "average")
    piv['numpy / scipy'] = piv['numpy'] / piv['scipy']
    piv['numpy-lower / scipy'] = piv['numpy-lower'] / piv['scipy']
    piv['onnx-py / scipy'] = piv['onnx-py'] / piv['scipy']
    piv['onnx-rt / scipy'] = piv['onnx-rt'] / piv['scipy']
    piv = piv[['numpy / scipy', 'numpy-lower / scipy', 
               'onnx-py / scipy', 'onnx-rt / scipy']]
    piv.plot(ax=ax[i], logy=True, logx=True)
    ax[i].plot([0, max(piv.index)], [1, 1], '--', color='black')
    ax[i].plot([0, max(piv.index)], [10, 10], '--', color='black')
    ax[i].set_title("ncol=%d" % ncol)
ax;


# ## Test with a new operator CDist
# 
# The final question is: *should we introduce a new operator into [ONNX specifications](https://github.com/onnx/onnx/blob/master/docs/Operators.md)?* The function [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) is not necessarily often used for a big number of observations as the square matrix it produces will even bigger. It seems reasonable. We showed that a python runtime based on *numpy* would not help, the implementation must be done in C++ or directly used the *scipy* version. The experiment was done with a [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html). The following section tests with and without a new operator ``CDist`` reusing *scipy* implementation.

# In[34]:


import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, __ = train_test_split(X, y, random_state=12)
clr = GaussianProcessRegressor(ExpSineSquared(), alpha=20.)
clr.fit(X_train, y_train)

model_def = to_onnx(clr, X_train)

get_ipython().run_line_magic('onnxview', 'model_def -r 1')


# In[35]:


model_def_cdist = to_onnx(clr, X_train,
                          options={GaussianProcessRegressor: {'optim': 'cdist'}})
get_ipython().run_line_magic('onnxview', 'model_def_cdist')


# In[36]:


oinf = OnnxInference(model_def)
oinf_cdist = OnnxInference(model_def_cdist)


# In[37]:


get_ipython().run_line_magic('timeit', "oinf.run({'X': X_test})")


# In[38]:


get_ipython().run_line_magic('timeit', "oinf_cdist.run({'X': X_test})")


# In[39]:


oinfrt = OnnxInference(model_def, runtime="onnxruntime1")
oinfrt_cdist = OnnxInference(model_def_cdist)


# In[40]:


get_ipython().run_line_magic('timeit', "oinfrt_cdist.run({'X': X_test})")


# It is 10 times faster for this dataset so it is worth it. For bigger datasets, we should expect a lower gain but still significant.

# In[41]: