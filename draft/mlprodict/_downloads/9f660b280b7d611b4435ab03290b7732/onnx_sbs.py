#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # ONNX side by side
# 
# The notebook compares two runtimes for the same ONNX and looks into differences at each step of the graph.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## The ONNX model
# 
# We convert kernel function used in [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html). First some values to use for testing.

# In[4]:


import numpy
import pandas
from io import StringIO

Xtest = pandas.read_csv(StringIO("""
1.000000000000000000e+02,1.061277971307766705e+02,1.472195004809226493e+00,2.307125069497626552e-02,4.539948095743629591e-02,2.855191098141335870e-01
1.000000000000000000e+02,9.417031896832908444e+01,1.249743892709246573e+00,2.370416174339620707e-02,2.613847280316268853e-02,5.097165413593484073e-01
1.000000000000000000e+02,9.305231488674536422e+01,1.795726729335217264e+00,2.473274733802270642e-02,1.349765645107412620e-02,9.410288840541443378e-02
1.000000000000000000e+02,7.411264142156210255e+01,1.747723020195752319e+00,1.559695663417645997e-02,4.230394035515055301e-02,2.225492746314280956e-01
1.000000000000000000e+02,9.326006195761877393e+01,1.738860294343326229e+00,2.280160135767652502e-02,4.883335335161764074e-02,2.806808409247734115e-01
1.000000000000000000e+02,8.341529291866362428e+01,5.119682123742423929e-01,2.488795768635816003e-02,4.887573336092913834e-02,1.673462179673477768e-01
1.000000000000000000e+02,1.182436477919874562e+02,1.733516391831658954e+00,1.533520930349476820e-02,3.131213519485807895e-02,1.955345358785769427e-01
1.000000000000000000e+02,1.228982583299257101e+02,1.115599996405831629e+00,1.929354155079938959e-02,3.056996308544096715e-03,1.197052763998271013e-01
1.000000000000000000e+02,1.160303269386108838e+02,1.018627021014927303e+00,2.248784981616459844e-02,2.688111547114307651e-02,3.326105131778724355e-01
1.000000000000000000e+02,1.163414374640396005e+02,6.644299545804077667e-01,1.508088417713602906e-02,4.451836657613789106e-02,3.245643044204808425e-01
""".strip("\n\r ")), header=None).values


# Then the kernel.

# In[5]:


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK, Sum

ker = Sum(
    CK(0.1, (1e-3, 1e3)) * RBF(length_scale=10,
                               length_scale_bounds=(1e-3, 1e3)),
    CK(0.1, (1e-3, 1e3)) * RBF(length_scale=1,
                               length_scale_bounds=(1e-3, 1e3))
)

ker


# In[6]:


ker(Xtest)


# ## Conversion to ONNX
# 
# The function is not an operator, the function to use is specific to this usage.

# In[7]:


from skl2onnx.operator_converters.gaussian_process import convert_kernel
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx.algebra.onnx_ops import OnnxIdentity
onnx_op = convert_kernel(ker, 'X', output_names=['final_after_op_Add'],
                         dtype=numpy.float32, op_version=12)
onnx_op = OnnxIdentity(onnx_op, output_names=['Y'], op_version=12)
model_onnx = model_onnx = onnx_op.to_onnx(
                inputs=[('X', FloatTensorType([None, None]))],
                target_opset=12)

with open("model_onnx.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())


# ``[('X', FloatTensorType([None, None]))]`` means the function applies on every tensor whatever its dimension is.

# In[8]:


get_ipython().run_line_magic('onnxview', 'model_onnx')


# In[9]:


from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx
# line needed when onnx is more recent than onnxruntime
model_onnx.ir_version = get_ir_version_from_onnx()
pyrun = OnnxInference(model_onnx, inplace=False)
rtrun = OnnxInference(model_onnx, runtime="onnxruntime1")


# In[10]:


pyres = pyrun.run({'X': Xtest.astype(numpy.float32)})
pyres


# In[11]:


rtres = rtrun.run({'X': Xtest.astype(numpy.float32)})
rtres


# In[12]:


from mlprodict.onnxrt.validate.validate_difference import measure_relative_difference
measure_relative_difference(pyres['Y'], rtres['Y'])


# The last runtime uses the same runtime but with double instead of floats.

# In[13]:


onnx_op_64 = convert_kernel(ker, 'X', output_names=['final_after_op_Add'],
                            dtype=numpy.float64, op_version=12)
onnx_op_64 = OnnxIdentity(onnx_op_64, output_names=['Y'], op_version=12)
model_onnx_64 = onnx_op_64.to_onnx(
                    inputs=[('X', DoubleTensorType([None, None]))],
                    target_opset=12)


# In[14]:


pyrun64 = OnnxInference(model_onnx_64, runtime="python", inplace=False)
pyres64 = pyrun64.run({'X': Xtest.astype(numpy.float64)})
measure_relative_difference(pyres['Y'], pyres64['Y'])


# ## Side by side
# 
# We run every node independently and we compare the output at each step.

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


from mlprodict.onnxrt.validate.side_by_side import side_by_side_by_values
from pandas import DataFrame

def run_sbs(r1, r2, r3, x):
    sbs = side_by_side_by_values([r1, r2, r3], 
                                 inputs=[
                                     {'X': x.astype(numpy.float32)},
                                     {'X': x.astype(numpy.float32)},
                                     {'X': x.astype(numpy.float64)},
                                 ])
    df = DataFrame(sbs)
    dfd = df.drop(['value[0]', 'value[1]', 'value[2]'], axis=1).copy()
    dfd.loc[dfd.cmp == 'ERROR->=inf', 'v[1]'] = 10
    return dfd, sbs

dfd, _ = run_sbs(pyrun, rtrun, pyrun64, Xtest)
dfd


# In[17]:


ax = dfd[['name', 'v[2]']].iloc[1:].set_index('name').plot(kind='bar', figsize=(14,4), logy=True)
ax.set_title("relative difference for each output between python and onnxruntime");


# Let's try for other inputs.

# In[18]:


import warnings
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    values = [4, 6, 8, 12]
    fig, ax = plt.subplots(len(values), 2, figsize=(14, len(values) * 4))

    for i, d in enumerate(values):
        for j, dim in enumerate([3, 8]):
            mat = numpy.random.rand(d, dim)
            dfd, _ = run_sbs(pyrun, rtrun, pyrun64, mat)
            dfd[['name', 'v[1]']].iloc[1:].set_index('name').plot(
                kind='bar', figsize=(14,4), logy=True, ax=ax[i, j])
            ax[i, j].set_title("abs diff input shape {}".format(mat.shape))
            if i < len(values) - 1:
                for xlabel_i in ax[i, j].get_xticklabels():
                    xlabel_i.set_visible(False)


# ## Further analysis
# 
# If there is one issue, we can create a simple graph to test. We consider ``Y = A + B`` where *A* and *B* have the following name in the *ONNX* graph:

# In[19]:


node = pyrun.sequence_[-2].onnx_node
final_inputs = list(node.input)
final_inputs


# In[20]:


_, sbs = run_sbs(pyrun, rtrun, pyrun64, Xtest)

names = final_inputs + ['Y']
values = {}
for row in sbs:
    if row.get('name', '#') not in names:
        continue
    name = row['name']
    values[name] = [row["value[%d]" % i] for i in range(3)]

list(values.keys())


# Let's check.

# In[21]:


for name in names:
    if name not in values:
        raise Exception("Unable to find '{}' in\n{}".format(
            name, [_.get('name', "?") for _ in sbs]))

a, b, c = names
for i in [0, 1, 2]:
    A = values[a][i]
    B = values[b][i]
    Y = values[c][i]
    diff = Y - (A + B)
    dabs = numpy.max(numpy.abs(diff))
    print(i, diff.dtype, dabs)


# If the second runtime has issue, we can create a single node to check something.

# In[22]:


from skl2onnx.algebra.onnx_ops import OnnxAdd
onnx_add = OnnxAdd('X1', 'X2', output_names=['Y'], op_version=12)
add_onnx = onnx_add.to_onnx({'X1': A, 'X2': B}, target_opset=12)


# In[23]:


add_onnx.ir_version = get_ir_version_from_onnx()
pyrun_add = OnnxInference(add_onnx, inplace=False)
rtrun_add = OnnxInference(add_onnx, runtime="onnxruntime1")


# In[24]:


res1 = pyrun_add.run({'X1': A, 'X2': B})
res2 = rtrun_add.run({'X1': A, 'X2': B})


# In[25]:


measure_relative_difference(res1['Y'], res2['Y'])


# No mistake here.

# ## onnxruntime

# In[26]:


from onnxruntime import InferenceSession, RunOptions, SessionOptions
opt = SessionOptions()
opt.enable_mem_pattern = True
opt.enable_cpu_mem_arena = True
sess = InferenceSession(model_onnx.SerializeToString(), opt)
sess


# In[27]:


res = sess.run(None, {'X': Xtest.astype(numpy.float32)})[0]
measure_relative_difference(pyres['Y'], res)


# In[28]:


res = sess.run(None, {'X': Xtest.astype(numpy.float32)})[0]
measure_relative_difference(pyres['Y'], res)


# ## Side by side for MLPRegressor

# In[29]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = MLPRegressor()
clr.fit(X_train, y_train)


# In[30]:


from mlprodict.onnx_conv import to_onnx
onx = to_onnx(clr, X_train.astype(numpy.float32), target_opset=12)


# In[31]:


onx.ir_version = get_ir_version_from_onnx()
pyrun = OnnxInference(onx, runtime="python", inplace=False)
rtrun = OnnxInference(onx, runtime="onnxruntime1")
rt_partial_run = OnnxInference(onx, runtime="onnxruntime2")
dfd, _ = run_sbs(rtrun, rt_partial_run, pyrun, X_test)
dfd


# In[32]:


get_ipython().run_line_magic('onnxview', 'onx')


# In[33]: