#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Infer operator computation cost
# 
# This notebooks explores a way to predict the cost of operator Transpose based on some features.

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## ONNX graph and measures

# In[4]:


import numpy
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxTranspose


def create_onnx_graph(perm=(0, 1, 2, 3), target_opset=14):
    tr = OnnxTranspose('X', perm=perm, output_names=['Y'], op_version=target_opset)
    return tr.to_onnx({'X': FloatTensorType([None] * len(perm))})


onx = create_onnx_graph()

get_ipython().run_line_magic('onnxview', 'onx')


# In[5]:


from mlprodict.onnxrt import OnnxInference

onx = create_onnx_graph(perm=(1, 0, 3, 2))
oinf = OnnxInference(onx)
inputs = {'X': numpy.full((5, 6, 7, 8), 1, dtype=numpy.float32)}
res = oinf.run(inputs)['Y']
res.shape


# In[6]:


from onnxruntime import InferenceSession
sess = InferenceSession(onx.SerializeToString())
res = sess.run(None, inputs)[0]
res.shape


# In[7]:


from cpyquickhelper.numbers.speed_measure import measure_time

def measure_time_onnx(sess, X, number=50, repeat=30):
    inputs = {'X': X}
    return measure_time(lambda: sess.run(None, inputs), context=dict(sess=sess, inputs=inputs),
                        div_by_number=True, number=number, repeat=repeat)

X = numpy.random.random((3, 224, 224, 4)).astype(numpy.float32)
measure_time_onnx(sess, X)


# ## Simulation to build a database

# ### Many dimensions, many permutations

# In[8]:


from itertools import permutations
from tqdm import tqdm
from pandas import DataFrame


def process_shape(shape, rnd=False, number=50, repeat=30, bar=True):
    X = numpy.random.random(shape).astype(numpy.float32)
    obs = []
    perms = list(permutations(list(range(len(X.shape)))))
    baseline = None
    itergen = perms if (rnd or not bar) else tqdm(perms)
    for perm in itergen:
        if baseline is not None and rnd:
            if random.randint(0, 4) != 0:
                continue
        onx = create_onnx_graph(perm=perm)
        sess = InferenceSession(onx.SerializeToString())
        res = measure_time_onnx(sess, X, number=number, repeat=repeat)
        res['perm'] = perm
        res['shape'] = shape
        if baseline is None:
            baseline = res
        res["ratio"] = res["average"] / baseline["average"]
        res['dim'] = len(shape)
        obs.append(res)
    return DataFrame(obs).sort_values('average')

dfs = []
df = process_shape((12, 13, 15, 18))
dfs.append(df)
df


# In[9]:


df = process_shape((43, 44, 45))
dfs.append(df)
df


# In[10]:


df = process_shape((3, 244, 244))
dfs.append(df)
df


# In[11]:


df = process_shape((3, 244, 244, 1))
dfs.append(df)
df


# In[12]:


df = process_shape((1, 244, 244, 3))
dfs.append(df)
df


# In[13]:


df = process_shape((3, 244, 244, 3), number=15, repeat=15)
dfs.append(df)
df


# In[14]:


df = process_shape((3, 244, 244, 6), number=15, repeat=15)
dfs.append(df)
df


# ### Random cases

# In[15]:


import random

if False:  # comment out for more training data
    for i in tqdm(range(0, 30)):
        dim = random.randint(3, 5)
        total = 1e8
        while total > 1e6 or total < 0:
            if dim == 3:
                shape = [random.randint(3, 64), random.randint(3, 224), random.randint(3, 64)]
            elif dim == 4:
                shape = (
                    [random.randint(3, 8)] + 
                    [random.randint(16, 224) for d in range(2)] +
                    [random.randint(16, 64)])
            elif dim == 5:
                shape = (
                    [random.randint(3, 8)] + 
                    [random.randint(16, 32) for d in range(3)] +
                    [random.randint(16, 64)])
            else:
                raise NotImplementedError()
            ashape = numpy.array(shape, dtype=numpy.float64)
            total = numpy.prod(ashape)

        if total > 1000000:
            number, repeat = 2, 2
        elif total > 800000:
            number, repeat = 3, 3
        elif total > 500000:
            number, repeat = 5, 5
        elif total > 200000:
            number, repeat = 7, 7
        else:
            number, repeat = 10, 10

        df = process_shape(tuple(shape), number=number, repeat=repeat, bar=False)
        dfs.append(df)

        for i in range(len(shape)):
            shape2 = shape.copy()
            shape2[i] = 1
            df = process_shape(tuple(shape), number=number, repeat=repeat, bar=False)
            dfs.append(df)
    
len(dfs)


# In[16]:


import pandas

data = pandas.concat(dfs, axis=0).reset_index(drop=True)
data.tail()


# In[17]:


data.shape


# In[18]:


data[['dim', 'shape', 'ratio']].groupby(['dim', 'shape']).agg({'ratio': [min, max, numpy.mean, numpy.median]})


# ## features
# 
# 

# ### Computing the features

# In[19]:


def _edit_distance(mot1, mot2):
    dist = {(-1, -1): 0}
    pred = {(-1, -1): None}
    if len(mot1) == 0:
        for j, d in enumerate(mot2):
            dist[-1, j] = dist[-1, j - 1] + 1
            pred[-1, j] = (-1, j - 1)
            dist[j, -1] = dist[j - 1, -1] + 1
            pred[j, -1] = (j - 1, -1)
    for i, c in enumerate(mot1):
        dist[i, -1] = dist[i - 1, -1] + 1
        pred[i, -1] = (i - 1, -1)
        dist[-1, i] = dist[-1, i - 1] + 1
        pred[-1, i] = (-1, i - 1)
        for j, d in enumerate(mot2):
            opt = []
            if (i - 1, j) in dist:
                x = dist[i - 1, j] + 1
                opt.append((x, (i - 1, j)))
            if (i, j - 1) in dist:
                x = dist[i, j - 1] + 1
                opt.append((x, (i, j - 1)))
            if (i - 1, j - 1) in dist:
                x = dist[i - 1, j - 1] + (1 if c != d else 0)
                opt.append((x, (i - 1, j - 1)))
            mi = min(opt)
            dist[i, j] = mi[0]
            pred[i, j] = mi[1]

    return dist[len(mot1) - 1, len(mot2) - 1]

_edit_distance("abdc", "cbda")


# In[20]:


_edit_distance((0, 1, 2, 3), (0, 2, 1, 3))


# In[21]:


from math import log


def _is_rotation(perm):
    t = tuple(perm)
    c = list(range(len(perm)))
    for i in range(len(c)):
        for k in range(len(c)):
            c[k] = (k + i) % len(c)
        if t == tuple(c):
            return True
    return False


def _relu(x, origin=0):
    return origin if x < origin else x


def compute_features(shape, perm):    
    total = numpy.prod(numpy.array(shape, dtype=numpy.int64))
    
    begin = 1
    dbegin = 0
    for i, p in enumerate(perm):
        if p != i:
            break
        dbegin += 1
        begin *= shape[i]
        
    end = 1
    dend = 0
    for i in range(len(perm)-1, -1, -1):
        if perm[i] != i:
            break
        dend += 1
        end *= shape[i]
    
    dis_cont = 0
    for i in range(1, len(shape)):
        if perm[i] != perm[i-1] + 1:
            dis_cont += 1
    
    middle = max(1, int(total / (end * begin)))
    feat = dict(size=total, begin=begin, end=end, middle=middle,
                dim=len(shape), discont=dis_cont)

    for c in [16, 32]:
        feat["end%d" % c] = _relu(end, c)
    
    keys = list(feat)
    for k in keys:
        if k in {'dim', 'cpu', 'size'}:
            continue
        feat['r%s' % k] = float(feat[k] / total)
    
    for c in [2, 4, 8, 16, 32, 64]:
        feat["iend%d" % c] = float(end >= c)
        feat["ibegin%d" % c] = float(begin >= c)
    
    # feat['CST'] = 1
    feat['CST_'] = -1
    feat['dbegin'] = - dbegin
    feat['dend'] = - dend
    
    keys = list(feat)
    for k in keys:
        if k.startswith('end') or k.startswith('begin'):
            feat[k] = - feat[k]
        elif k.startswith('rend') or k.startswith('rbegin'):
            feat[k] = - feat[k]
        elif k.startswith('iend') or k.startswith('ibegin'):
            feat[k] = - feat[k]
        elif k == "rdiscont":
            feat[k] = - feat[k]

    idp = list(range(len(perm)))
    feat["rot"] = -1 if _is_rotation(perm) else 0
    feat["rev"] = 1 if perm == tuple(idp[::-1]) else 0
    feat["edit"] = _edit_distance(idp, perm)
    feat["redit"] = feat["edit"] / len(idp)
    return feat


compute_features((3, 5, 7), (0, 1, 2))


# In[22]:


compute_features((3, 5, 7), (2, 1, 0))


# In[23]:


compute_features((3, 5, 7), (1, 2, 0))


# ### Computing the features for all simulations

# In[24]:


def compute_features_dataframe(df):
    
    def merge(row):
        feat = compute_features(row['shape'], row['perm'])
        feat['yt'] = row['average']
        feat['yr'] = row['ratio']
        return feat
    
    rows = []
    for i in tqdm(range(df.shape[0])):
        rows.append(dict(shape=df.loc[i, "shape"], perm=df.loc[i, "perm"],
                         average=df.loc[i, "average"], ratio=df.loc[i, "ratio"]))
    obs = []
    for row in tqdm(rows):
        obs.append(merge(row))
    return DataFrame(obs)

fdata = compute_features_dataframe(data)
col_sort = list(sorted(fdata.columns))
fdata = fdata[col_sort]
fdata.tail()


# ### correlations

# In[25]:


fdata.corr()


# In[26]:


fdata.corr()['yt']


# We check the sign of the correlations of all features with *yt*. If it is positive, increasing the feature increases the processing time. We try to get only positive correlations. *end* is the flattened last dimensions left unchanged by the permutation. The bigger it is, the faster the transposition is. That's why the function computing all features multiplies this number by `-1` to get a feature positively correlated to the processing time. *end16* is equal to *end* when `end<-16` and `-16` when `end>=-16`. This is a simplification of the cost of moving data from memory to cache L1. This cost is linear when the data to move is big enough, but almost constant for small chunks.

# ## Linear regression
# 
# We choose a linear regression because the prediction are not limited. The training set does not include all configuration and surely does not include all possible high value the model may have to predict.
# 
# The goal is not necessarily to predict the fastest permutation but to predict the processing time as the goal is to find the best combination of transpositions in a ONNX graph (einsum). The final goal is to predict which graphs optimizes a series of transpositions.
# 
# The target could be the processing time or the logarithm of this time. However, making mistakes on small times is not an issue but errors on high processing time is not a good thing.
# 
# We could also try to predict a ratio *transposition time /copy time* but it still gives more important to small matrix size.  
# 
# Many variables are correlated. Variables should be selected.

# ### Dataset

# In[27]:


X = fdata.drop(["yt", "yr"], axis=1)
x_names = list(X.columns)
yt = fdata['yt'] * 1000


# In[28]:


numpy.mean(yt)


# ### Simple model 

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

pipe = make_pipeline(StandardScaler(with_mean=False), LinearRegression(fit_intercept=False))
pipe.fit(X, yt)
model = pipe.steps[1][1]
coef = {k: v for k, v in zip(X.columns, model.coef_)}
coef['name'] = 'reg'
coef['intercept_'] = model.intercept_
pred = numpy.maximum(pipe.predict(X), 0)
coef['r2'] = r2_score(yt, pred)
coef['mae'] = mean_absolute_error(yt, pred)
coef['model'] = pipe
coefs = [coef]
coef["r2"], coef['mae']


# In[30]:


df = DataFrame([(k, v) for k, v in coef.items() if k not in {'name', 'model'}],
                columns=["feature", "value"]).set_index("feature")
df.plot(kind="bar", figsize=(14, 2));


# In[31]:


df


# Coefficients associated to features *end*, *end16* are almost opposed and it would better to get a model which keeps only one.

# ### Quantile Regression

# In[32]:


from mlinsights.mlmodel import QuantileLinearRegression
pipe = make_pipeline(StandardScaler(with_mean=False), QuantileLinearRegression(fit_intercept=False))
pipe.fit(X, yt)
model = pipe.steps[1][1]
coef = {k: v for k, v in zip(X.columns, model.coef_)}
coef['name'] = 'med'
coef['intercept_'] = model.intercept_
pred = numpy.maximum(pipe.predict(X), 0)
coef['r2'] = r2_score(yt, pred)
coef['mae'] = mean_absolute_error(yt, pred)
coef['model'] = pipe
coefs.append(coef)
coef["r2"], coef['mae']


# In[33]:


DataFrame(coef.items(), columns=["feature", "value"]).set_index("feature")


# ### Lasso
# 
# To select features.

# In[34]:


from sklearn.linear_model import Lasso

scores = []
models = []
for a in tqdm([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2.]):
    alpha = a * 1.
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        Lasso(alpha=alpha, fit_intercept=False, max_iter=5000))
    pipe.fit(X, yt)
    pred = numpy.maximum(pipe.predict(X), 0)
    model = pipe.steps[1][1]
    scores.append(dict(r2=r2_score(yt, pred), mae=mean_absolute_error(yt, pred),
                       alpha=alpha, null=(numpy.abs(model.coef_) < 1e-6).sum(),
                       n=len(model.coef_)))
    models.append(pipe)
    if alpha >= 0.01 and alpha <= 0.2:
        coef = {k: v for k, v in zip(X.columns, pipe.steps[1][1].coef_)}
        coef['name'] = "Lasso-%f" % alpha
        coef['model'] = pipe
        coef['r2'] = r2_score(yt, pred)
        coef['mae'] = mean_absolute_error(yt, pred)
        coefs.append(coef)
    
DataFrame(scores)


# In[35]:


coef = {k: v for k, v in zip(X.columns, models[1].steps[1][1].coef_)}
df = DataFrame(coef.items(), columns=["feature", "value"]).set_index("feature")
df.plot(kind="bar", figsize=(14, 2), title="alpha=%f" % scores[1]["alpha"]);


# In[36]:


coef = {k: v for k, v in zip(X.columns, models[2].steps[1][1].coef_)}
df = DataFrame(coef.items(), columns=["feature", "value"]).set_index("feature")
df.plot(kind="bar", figsize=(14, 2), title="alpha=%f" % scores[2]["alpha"]);


# ### Linear regression with positive weights

# In[37]:


pipe = make_pipeline(StandardScaler(with_mean=False), LinearRegression(positive=True, fit_intercept=False))
pipe.fit(X, yt)
model = pipe.steps[1][1]
coef = {k: v for k, v in zip(X.columns, model.coef_)}
coef['name'] = 'pos'
coef['intercept_'] = model.intercept_
pred = numpy.maximum(pipe.predict(X), 0)
coef['r2'] = r2_score(yt, pred)
coef['mae'] = mean_absolute_error(yt, pred)
coef['model'] = pipe
coefs.append(coef)
coef["r2"], coef['mae']


# In[38]:


coef = {k: v for k, v in zip(X.columns, pipe.steps[1][1].coef_)}
df = DataFrame(coef.items(), columns=["feature", "value"]).set_index("feature")
df.plot(kind="bar", figsize=(14, 2), title="positive");


# ### Quantile regression with positive weights

# In[39]:


pipe = make_pipeline(StandardScaler(with_mean=False), QuantileLinearRegression(positive=True, fit_intercept=False))
pipe.fit(X, yt)
model = pipe.steps[1][1]
coef = {k: v for k, v in zip(X.columns, model.coef_)}
coef['name'] = 'medpos'
coef['intercept_'] = model.intercept_
pred = numpy.maximum(pipe.predict(X), 0)
coef['r2'] = r2_score(yt, pred)
coef['mae'] = mean_absolute_error(yt, pred)
coef['model'] = pipe
coefs.append(coef)
coef["r2"], coef['mae']


# In[40]:


coef = {k: v for k, v in zip(X.columns, pipe.steps[1][1].coef_)}
df = DataFrame(coef.items(), columns=["feature", "value"]).set_index("feature")
df.plot(kind="bar", figsize=(14, 2), title="positive");


# ### Summary

# In[41]:


dfcoef = DataFrame(coefs)
dfcoef[::-1].T


# In[42]:


dfcoef[["name", "r2", "mae"]].set_index('name').plot(kind="bar", title="performance accross models");


# In[43]:


import matplotlib.pyplot as plt

dfp = dfcoef.drop(['name', 'model'], axis=1).T.drop([0, 1], axis=1).copy()
dfp.columns = dfcoef['name'][2:]
ax = dfp.plot(figsize=(14, 4), kind="line")
ax.set_xticks(numpy.arange(0, dfp.shape[0]))
ax.set_xticklabels(dfp.index)
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right');


# ## Investigation

# In[44]:


data_err = data.drop(["context_size", "repeat"], axis=1).copy()
data_err['predict'] = numpy.maximum(coefs[0]['model'].predict(X), 0) / 1000
data_err['err'] = (data_err['predict'] - data_err['average'])
data_err['abserr'] = numpy.abs(data_err['predict'] - data_err['average'])
data_err['rel'] = (data_err['predict'] - data_err['average']) / data_err['average']
s = data_err.sort_values('abserr')
pandas.concat([s.head(n=10), s.tail(n=10)])


# All big errors are negative. The model seems to give a lower value for all big errors. These errors may be outliers, the processor was busy doing something else at that time.

# In[45]:


s = data_err.sort_values('predict')
pandas.concat([s.head(n=10), s.tail(n=10)])


# ### Correlation between predictors

# In[46]:


cc = DataFrame(dict([(c['name'], numpy.maximum(c['model'].predict(X), 0)) for c in coefs]))
cc['yt'] = yt
cc


# In[47]:


cc.corr()


# ## Standalone predictions

# In[48]:


def get_coef(pipe, names):
    c1 = pipe.steps[0][-1].scale_
    c2 = pipe.steps[1][-1].coef_
    return dict(zip(names, c2 / c1))


get_coef(coefs[-1]["model"], X.columns)


# In[49]:


def predict(coefs, shape, perm):
    feat = compute_features(shape, perm)
    res = 0
    for k, v in feat.items():
        res += v * coefs[k]
    return res / 1000


def predict_model(model, shape, perm, names):
    feat = compute_features(shape, perm)
    a = numpy.zeros((1, len(names)), dtype=numpy.float64)
    for i, n in enumerate(names):
        a[0, i] = feat[n]
    return model.predict(a) / 1000
    

coef = get_coef(coefs[-1]["model"], X.columns)
(predict(coef, (3, 224, 224, 6), (3, 0, 1, 2)), 
 predict_model(coefs[-1]["model"], (3, 224, 224, 6), (3, 0, 1, 2), X.columns))


# In[50]: