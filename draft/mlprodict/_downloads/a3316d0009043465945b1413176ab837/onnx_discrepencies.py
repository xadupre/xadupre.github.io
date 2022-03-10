#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Discrepencies with ONNX
# 
# The notebook shows one example where the conversion leads with discrepencies if default options are used. It converts a pipeline with two steps, a scaler followed by a tree.

# The bug this notebook is tracking does not always appear, it has a better chance to happen with integer features but that's not always the case. The notebook must be run again in that case.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data and first model
# 
# We take a random datasets with mostly integers.

# In[3]:


import math
import numpy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(10000, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

Xi_train, yi_train = X_train.copy(), y_train.copy()
Xi_test, yi_test = X_test.copy(), y_test.copy()
for i in range(X.shape[1]):
    Xi_train[:, i] = (Xi_train[:, i] * math.pi * 2 ** i).astype(numpy.int64)
    Xi_test[:, i] = (Xi_test[:, i] * math.pi * 2 ** i).astype(numpy.int64)


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

max_depth = 10

model = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model.fit(Xi_train, yi_train)


# In[5]:


model.predict(Xi_test[:5])


# Other models:

# In[6]:


model2 = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])
model3 = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=3))
])


models = [
    ('bug', Xi_test.astype(numpy.float32), model),
    ('no scaler', Xi_test.astype(numpy.float32), 
     DecisionTreeRegressor(max_depth=max_depth).fit(Xi_train, yi_train)),
    ('float', X_test.astype(numpy.float32),
     model2.fit(X_train, y_train)),
    ('max_depth=3', X_test.astype(numpy.float32),
     model3.fit(X_train, y_train))
]


# ## Conversion to ONNX

# In[7]:


import numpy
from mlprodict.onnx_conv import to_onnx

onx = to_onnx(model, X_train[:1].astype(numpy.float32))


# In[8]:


from mlprodict.onnxrt import OnnxInference

oinfpy = OnnxInference(onx, runtime="python_compiled")
print(oinfpy)


# In[9]:


import pandas

X32 = Xi_test.astype(numpy.float32)
y_skl = model.predict(X32)

obs = [dict(runtime='sklearn', diff=0)]
for runtime in ['python', 'python_compiled', 'onnxruntime1']:
    oinf = OnnxInference(onx, runtime=runtime)
    y_onx = oinf.run({'X': X32})['variable']
    delta = numpy.abs(y_skl - y_onx.ravel())
    am = delta.argmax()
    obs.append(dict(runtime=runtime, diff=delta.max()))
    obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
    obs[0]['v[%d]' % am] = y_skl.ravel()[am]

pandas.DataFrame(obs)


# The pipeline shows huge discrepencies. They appear for a pipeline *StandardScaler* + *DecisionTreeRegressor* applied in integer features. They disappear if floats are used, or if the scaler is removed. The bug also disappear if the tree is not big enough (max_depth=4 instread of 5).

# In[10]:


obs = [dict(runtime='sklearn', diff=0, name='sklearn')]
for name, x32, mod in models:
    for runtime in ['python', 'python_compiled', 'onnxruntime1']:
        lonx = to_onnx(mod, x32[:1])
        loinf = OnnxInference(lonx, runtime=runtime)
        y_skl = mod.predict(X32)
        y_onx = loinf.run({'X': X32})['variable']
        delta = numpy.abs(y_skl - y_onx.ravel())
        am = delta.argmax()
        obs.append(dict(runtime=runtime, diff=delta.max(), name=name))
        obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
        obs[0]['v[%d]' % am] = y_skl.ravel()[am]

df = pandas.DataFrame(obs)
df


# In[11]:


df.pivot("runtime", "name", "diff")


# ## Other way to convert
# 
# ONNX does not support double for TreeEnsembleRegressor but that a new operator TreeEnsembleRegressorDouble was implemented into *mlprodict*. We need to update the conversion.

# In[12]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[13]:


onx32 = to_onnx(model, X_train[:1].astype(numpy.float32))
onx64 = to_onnx(model, X_train[:1].astype(numpy.float64), 
                rewrite_ops=True)
get_ipython().run_line_magic('onnxview', 'onx64')


# In[14]:


X32 = Xi_test.astype(numpy.float32)
X64 = Xi_test.astype(numpy.float64)

obs = [dict(runtime='sklearn', diff=0)]
for runtime in ['python', 'python_compiled', 'onnxruntime1']:
    for name, onx, xr in [('float', onx32, X32), ('double', onx64, X64)]:
        try:
            oinf = OnnxInference(onx, runtime=runtime)
        except Exception as e:
            obs.append(dict(runtime=runtime, error=str(e), real=name))
            continue
        y_skl = model.predict(xr)
        y_onx = oinf.run({'X': xr})['variable']
        delta = numpy.abs(y_skl - y_onx.ravel())
        am = delta.argmax()
        obs.append(dict(runtime=runtime, diff=delta.max(), real=name))
        obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
        obs[0]['v[%d]' % am] = y_skl.ravel()[am]

pandas.DataFrame(obs)


# We see that the use of double removes the discrepencies.

# ## OnnxPipeline
# 
# Another way to reduce the number of discrepencies is to use a pipeline which converts every steps into ONNX before training the next one. That way, every steps is either trained on the inputs, either trained on the outputs produced by ONNX. Let's see how it works.

# In[15]:


from mlprodict.sklapi import OnnxPipeline

model_onx = OnnxPipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])
model_onx.fit(Xi_train, yi_train)


# We see that the first steps was replaced by an object *OnnxTransformer* which wraps an ONNX file into a transformer following the *scikit-learn* API. The initial steps are still available.

# In[16]:


model_onx.raw_steps_


# In[17]:


models = [
    ('bug', Xi_test.astype(numpy.float32), model),
    ('OnnxPipeline', Xi_test.astype(numpy.float32), model_onx),
]


# In[18]:


obs = [dict(runtime='sklearn', diff=0, name='sklearn')]
for name, x32, mod in models:
    for runtime in ['python', 'python_compiled', 'onnxruntime1']:
        lonx = to_onnx(mod, x32[:1])
        loinf = OnnxInference(lonx, runtime=runtime)
        y_skl = model_onx.predict(X32)  # model_onx is the new baseline
        y_onx = loinf.run({'X': X32})['variable']
        delta = numpy.abs(y_skl - y_onx.ravel())
        am = delta.argmax()
        obs.append(dict(runtime=runtime, diff=delta.max(), name=name))
        obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
        obs[0]['v[%d]' % am] = y_skl.ravel()[am]

df = pandas.DataFrame(obs)
df


# Training the next steps based on ONNX outputs is better. This is not completely satisfactory... Let's check the accuracy.

# In[19]:


model.score(Xi_test, yi_test), model_onx.score(Xi_test, yi_test)


# Pretty close.

# ## Final explanation: StandardScalerFloat
# 
# We proposed two ways to have an ONNX pipeline which produces the same prediction as *scikit-learn*. Let's now replace the StandardScaler by a new one which outputs float and not double. It turns out that class *StandardScaler* computes ``X /= self.scale_`` but ONNX does ``X *= self.scale_inv_``. We need to implement this exact same operator with float32 to remove all discrepencies.

# In[20]:


class StandardScalerFloat(StandardScaler):
    
    def __init__(self, with_mean=True, with_std=True):
        StandardScaler.__init__(self, with_mean=with_mean, with_std=with_std)
    
    def fit(self, X, y=None):
        StandardScaler.fit(self, X, y)
        if self.scale_ is not None:
            self.scale_inv_ = (1. / self.scale_).astype(numpy.float32)
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X *= self.scale_inv_
        return X

    
model_float = Pipeline([
    ('scaler', StandardScalerFloat()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model_float.fit(Xi_train.astype(numpy.float32), yi_train.astype(numpy.float32))


# In[21]:


try:
    onx_float = to_onnx(model_float, Xi_test[:1].astype(numpy.float))
except RuntimeError as e:
    print(e)


# We need to register a new converter so that *sklearn-onnx* knows how to convert the new scaler. We reuse the existing converters.

# In[22]:


from skl2onnx import update_registered_converter
from skl2onnx.operator_converters.scaler_op import convert_sklearn_scaler
from skl2onnx.shape_calculators.scaler import calculate_sklearn_scaler_output_shapes


update_registered_converter(
    StandardScalerFloat, "SklearnStandardScalerFloat",
    calculate_sklearn_scaler_output_shapes,
    convert_sklearn_scaler,
    options={'div': ['std', 'div', 'div_cast']})


# In[23]:


models = [
    ('bug', Xi_test.astype(numpy.float32), model),
    ('FloatPipeline', Xi_test.astype(numpy.float32), model_float),
]


# In[24]:


obs = [dict(runtime='sklearn', diff=0, name='sklearn')]
for name, x32, mod in models:
    for runtime in ['python', 'python_compiled', 'onnxruntime1']:
        lonx = to_onnx(mod, x32[:1])
        loinf = OnnxInference(lonx, runtime=runtime)
        y_skl = model_float.predict(X32)  # we use model_float as a baseline
        y_onx = loinf.run({'X': X32})['variable']
        delta = numpy.abs(y_skl - y_onx.ravel())
        am = delta.argmax()
        obs.append(dict(runtime=runtime, diff=delta.max(), name=name))
        obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
        obs[0]['v[%d]' % am] = y_skl.ravel()[am]

df = pandas.DataFrame(obs)
df


# That means than the differences between ``float32(X / Y)`` and ``float32(X) * float32(1 / Y)`` are big enough to select a different path in the decision tree. ``float32(X) / float32(Y)`` and ``float32(X) * float32(1 / Y)`` are also different enough to trigger a different path. Let's illustrate that on example:

# In[25]:


a1 = numpy.random.randn(100, 2) * 10
a2 = a1.copy()
a2[:, 1] *= 1000
a3 = a1.copy()
a3[:, 0] *= 1000

for i, a in enumerate([a1, a2, a3]):
    a = a.astype(numpy.float32)
    max_diff32 = numpy.max([
        numpy.abs(numpy.float32(x[0]) / numpy.float32(x[1]) - 
            numpy.float32(x[0]) * (numpy.float32(1) / numpy.float32(x[1])))
        for x in a])
    max_diff64 = numpy.max([
        numpy.abs(numpy.float64(x[0]) / numpy.float64(x[1]) - 
            numpy.float64(x[0]) * (numpy.float64(1) / numpy.float64(x[1])))
        for x in a])
    print(i, max_diff32, max_diff64)


# The last random set shows very big differences, obviously big enough to trigger a different path in the graph. The difference for double could probably be significant in some cases, not enough on this example.

# ## Change the conversion with option *div*
# 
# Option ``'div'`` was added to the converter for *StandardScaler* to change the way the scaler is converted.

# In[26]:


model = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])
model.fit(Xi_train, yi_train)


# In[27]:


onx_std = to_onnx(model, Xi_train[:1].astype(numpy.float32))

get_ipython().run_line_magic('onnxview', 'onx_std')


# In[28]:


onx_div = to_onnx(model, Xi_train[:1].astype(numpy.float32),
                  options={StandardScaler: {'div': 'div'}})
get_ipython().run_line_magic('onnxview', 'onx_div')


# In[29]:


onx_div_cast = to_onnx(model, Xi_train[:1].astype(numpy.float32),
                       options={StandardScaler: {'div': 'div_cast'}})
get_ipython().run_line_magic('onnxview', 'onx_div_cast')


# The ONNX graph is different and using division. Let's measure the discrepencies.

# In[30]:


X32 = Xi_test.astype(numpy.float32)
X64 = Xi_test.astype(numpy.float64)
models = [('bug', model, onx_std),
          ('div', model, onx_div),
          ('div_cast', model, onx_div_cast),]

obs = [dict(runtime='sklearn', diff=0, name='sklearn')]
for name, mod, onx in models:
    for runtime in ['python', 'python_compiled', 'onnxruntime1']:
        oinf = OnnxInference(onx, runtime=runtime)
        y_skl32 = mod.predict(X32)
        y_skl64 = mod.predict(X64)
        y_onx = oinf.run({'X': X32})['variable']

        delta32 = numpy.abs(y_skl32 - y_onx.ravel())
        am32 = delta32.argmax()
        delta64 = numpy.abs(y_skl64 - y_onx.ravel())
        am64 = delta64.argmax()

        obs.append(dict(runtime=runtime, diff32=delta32.max(), 
                        diff64=delta64.max(), name=name))
        obs[0]['v32[%d]' % am32] = y_skl32.ravel()[am32]
        obs[0]['v64[%d]' % am64] = y_skl64.ravel()[am64]
        obs[-1]['v32[%d]' % am32] = y_onx.ravel()[am32]
        obs[-1]['v64[%d]' % am64] = y_onx.ravel()[am64]

df = pandas.DataFrame(obs)
df


# The only combination which works is the model converted with option *div_cast* (use of division in double precision), float input for ONNX, double input for *scikit-learn*.

# ## Explanation in practice
# 
# Based on previous sections, the following example buids a case where discreprencies are significant.

# In[31]:


std = StandardScaler()
std.fit(Xi_train)
xt32 = Xi_test.astype(numpy.float32)
xt64 = Xi_test.astype(numpy.float64)
pred = std.transform(xt32)


# In[32]:


from onnxruntime import InferenceSession

onx32 = to_onnx(std, Xi_train[:1].astype(numpy.float32))
sess32 = InferenceSession(onx32.SerializeToString())
got32 = sess32.run(0, {'X': xt32})[0]
d32 = numpy.max(numpy.abs(pred.ravel() - got32.ravel()))
d32


# In[33]:


oinf32 = OnnxInference(onx32.SerializeToString())
gotpy32 = oinf32.run({'X': xt32})['variable']
dpy32 = numpy.max(numpy.abs(pred.ravel() - gotpy32.ravel()))
dpy32


# We tried to cast float into double before applying the normalisation and to cast back into single float. It does not help much.

# In[34]:


onx64 = to_onnx(std, Xi_train[:1].astype(numpy.float32),
                options={id(std): {'div': 'div'}})        
sess64 = InferenceSession(onx64.SerializeToString())
got64 = sess64.run(0, {'X': xt32})[0]
d64 = numpy.max(numpy.abs(pred.ravel() - got64.ravel()))
d64


# Last experiment, we try to use double all along.

# In[35]:


from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph

onx64_2 = to_onnx(std, Xi_train[:1].astype(numpy.float64))
try:
    sess64_2 = InferenceSession(onx64_2.SerializeToString())
except InvalidGraph as e:
    print(e)


# *onnxruntime* does not support this. Let's switch to *mlprodict*.

# In[36]:


onx64_2 = to_onnx(std, Xi_train[:1].astype(numpy.float64))
sess64_2 = OnnxInference(onx64_2, runtime="python")
pred64 = std.transform(xt64)
got64_2 = sess64_2.run({'X': xt64})['variable']
d64_2 = numpy.max(numpy.abs(pred64.ravel() - got64_2.ravel()))
d64_2


# Differences are lower if every operator is done with double.

# ## Conclusion
# 
# Maybe the best option is just to introduce a transform which just cast inputs into floats.

# In[37]:


model1 = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model1.fit(Xi_train, yi_train)


# In[38]:


from skl2onnx.sklapi import CastTransformer

model2 = Pipeline([
    ('cast64', CastTransformer(dtype=numpy.float64)),
    ('scaler', StandardScaler()),
    ('cast', CastTransformer()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model2.fit(Xi_train, yi_train)


# In[39]:


X32 = Xi_test.astype(numpy.float32)
models = [('model1', model1, X32), ('model2', model2, X32)]
options = [('-', None),
           ('div_cast', {StandardScaler: {'div': 'div_cast'}})]

obs = [dict(runtime='sklearn', diff=0, name='model1'),
       dict(runtime='sklearn', diff=0, name='model2')]
for name, mod, x32 in models:
    for no, opts in options:
        onx = to_onnx(mod, Xi_train[:1].astype(numpy.float32),
                      options=opts)
        for runtime in ['python', 'python_compiled', 'onnxruntime1']:
            try:
                oinf = OnnxInference(onx, runtime=runtime)
            except Exception as e:
                obs.append(dict(runtime=runtime, err=str(e),
                                name=name, options=no))
                continue
                
            y_skl = mod.predict(x32)
            try:
                y_onx = oinf.run({'X': x32})['variable']
            except Exception as e:
                obs.append(dict(runtime=runtime, err=str(e),
                                name=name, options=no))
                continue

            delta = numpy.abs(y_skl - y_onx.ravel())
            am = delta.argmax()

            obs.append(dict(runtime=runtime, diff=delta.max(),
                            name=name, options=no))
            obs[-1]['v[%d]' % am] = y_onx.ravel()[am]
            if name == 'model1':
                obs[0]['v[%d]' % am] = y_skl.ravel()[am]
                obs[1]['v[%d]' % am] = model2.predict(Xi_test).ravel()[am]
            elif name == 'model2':
                obs[0]['v[%d]' % am] = model1.predict(Xi_test).ravel()[am]
                obs[1]['v[%d]' % am] = y_skl.ravel()[am]

df = pandas.DataFrame(obs)
df


# It seems to work that way.

# In[40]: