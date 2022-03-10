#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Einsum decomposition
# 
# This notebook shows a way to decompose [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) into a subset of operations (expand_dims, squeeze, transpose, extended matrix multiplication).

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# ## Operator explanation with equation `bac,cd,def=ebc`
# 
# The operator einsum takes an equation and some inputs. Every letter involved in the equation is a loop. Let's see on one example.

# In[3]:


import numpy

m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2)) + 10
m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2)) + 100
m3 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2)) + 1000

equation = "bac,cd,def->ebc"
truth = numpy.einsum(equation, m1, m2, m3)
truth


# This summation is equalent to:

# In[4]:


res = numpy.zeros((2, 2, 2))
for a in range(0, 2):
    for b in range(0, 2):
        for c in range(0, 2):
            for d in range(0, 2):
                for e in range(0, 2):
                    for f in range(0, 2):
                        res[e, b, c] += m1[b, a, c] * m2[c, d] * m3[d, e, f]
res


# Theoritically, this summation is in this case has a cost of $O(N^6)$. However this simple computation is usually much longer than using matrix multiplications along the path. $O(N^4)$ is the cost of the heaviest matrix multiplication in this case). But to do that, the equation needs to be decomposed into a sequence of matrix multiplications.

# ### Decomposition of `bac,cd,def=ebc`

# In[5]:


import numpy
from mlprodict.testing.einsum import (
    decompose_einsum_equation, apply_einsum_sequence)


# In[6]:


m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2)) + 10
m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2)) + 100
m3 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2)) + 1000


# In[7]:


seq = decompose_einsum_equation("bac,cd,def->ebc")


# In[8]:


from jyquickhelper import RenderJsDot
RenderJsDot(seq.to_dot(size=7))


# Then the result can be obtained as follows:

# In[9]:


apply_einsum_sequence(seq, m1, m2, m3)


# ### operator matmul
# 
# This operator can be used to represent either a multiplication, either a matrix multiplication but it applies only on arrays with the same number of dimensions. It can be broken into multiplication of matrix multiplication.

# In[10]:


seq_clean = decompose_einsum_equation("bac,cd,def->ebc", strategy='numpy', clean=True)
RenderJsDot(seq_clean.to_dot(size=7))


# Operator *transpose_mm* is a regular transposition, it takes two inputs but only tranposes the first input before returning it. Operator *batch_dot* is a matrix multiplication. It is left that way on purpose as it may be implemented with function dot or gemm. The operator distinguishes between 3 kind of axes: batch axes, kept axes, sum(mation) axes. It then reshapes both input matrices with 3D tensors, batch axis, row axis, column axis to use function [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html).

# ### ONNX
# 
# The previous graph can be converted into ONNX.

# In[11]:


onx = seq_clean.to_onnx("Y", "X1", "X2", "X3", dtype=numpy.float32)
# with open("einsum.onnx", "wb") as f:
#      f.write(onx.SerializeToString())
get_ipython().run_line_magic('onnxview', 'onx')


# In[12]:


from onnxruntime import InferenceSession
sess = InferenceSession(onx.SerializeToString())
sess.run(None, {'X1': m1.astype(numpy.float32), 
                'X2': m2.astype(numpy.float32), 
                'X3': m3.astype(numpy.float32)})[0]


# ### onnxruntime

# In[13]:


import onnx
from onnx import helper, numpy_helper
from onnxruntime import InferenceSession


def make_model1(equation):
    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid('', 13)],
        graph=helper.make_graph(
            name='einsum_test',
            inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, None)],
            outputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, None)],
            nodes=[
                helper.make_node("Einsum", ["X", "Y", "Z"], ["A"], equation=equation)
            ]
        )
    )
    return model


model = make_model1("bac,cd,def->ebc")
sess = InferenceSession(model.SerializeToString())


# In[14]:


sess.run(None, {'X': m1.astype(numpy.float32), 
                'Y': m2.astype(numpy.float32), 
                'Z': m3.astype(numpy.float32)})[0]


# ### Benchmark
# 
# It clearly shows the summation done with the basic algorithm is the slowest.

# In[15]:


from mlprodict.onnxrt.validate.validate_helper import measure_time
from tqdm import tqdm
from pandas import DataFrame


def raw_product(m1, m2, m3):
    N = m1.shape[0]
    res = numpy.zeros((N, N, N))
    for a in range(0, N):
        for b in range(0, N):
            for c in range(0, N):
                for d in range(0, N):
                    for e in range(0, N):
                        for f in range(0, N):
                            res[e, b, c] += m1[b, a, c] * m2[c, d] * m3[d, e, f]
    return res


def benchmark0(equation):
    sess = None
    sess2 = None
    seq = None 
    seq2 = None 

    results = []
    for N in tqdm([2, 3, 4, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]):
        m1 = numpy.random.randn(N, N, N)
        m2 = numpy.random.randn(N, N)
        m3 = numpy.random.randn(N, N, N)

        if seq is None:
            seq = decompose_einsum_equation(equation, clean=True)
        if seq2 is None:
            seq2 = decompose_einsum_equation(equation, clean=True, strategy='numpy')
        if sess is None:
            model = make_model1(equation)
            sess = InferenceSession(model.SerializeToString())
        if sess2 is None:
            onx = seq2.to_onnx("Y", "X1", "X2", "X3", dtype=numpy.float32)
            sess2 = InferenceSession(onx.SerializeToString())

        res = measure_time(lambda x: numpy.einsum(equation, *x, optimize=True),
                           [m1, m2, m3],
                           repeat=10, number=10)

        res['name'] = "numpy.einsum"
        res["N"] = N
        results.append(res)

        if N <= 4:
            res = measure_time(lambda x: raw_product(*x),
                               [m1, m2, m3],
                               repeat=10, number=10)
            res['name'] = "raw_product"
            res["N"] = N
            results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x),
                           [m1, m2, m3],
                           repeat=10, number=10)

        res['name'] = "custom_einsum"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x, matmul_impl="pyf"),
                           [m1, m2, m3],
                           repeat=10, number=10)
        res['name'] = "dec-matmul"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq2, *x, matmul_impl="pyf"),
                           [m1, m2, m3],
                           repeat=10, number=10)
        res['name'] = "dec-batch_dot"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: sess.run(None, {'X': x[0], 'Y': x[1], 'Z': x[2]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=10, number=10)
        res['name'] = "ort-einsum"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: sess2.run(None, {'X1': x[0], 'X2': x[1], 'X3': x[2]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=10, number=10)
        res['name'] = "ort-matmul"
        res["N"] = N
        results.append(res)    
    return DataFrame(results)

df = benchmark0("bac,cd,def->ebc")
df.tail()


# In[16]:


import matplotlib.pyplot as plt

piv = df.pivot("N", "name", "average")
piv2 = piv.copy()
np = piv["numpy.einsum"]
for c in piv2.columns:
    piv2[c] /= np
    
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
piv.plot(logy=True, logx=True, ax=ax[0])
ax[0].set_title("Benchmark einsum function\nbac,cd,def->ebc")
piv2.plot(logy=True, logx=True, ax=ax[1])
ax[1].set_title("Benchmark einsum function\n(ratio, baseline=numpy)");


# Version `dec-matmul` is an implementation based on the decomposition of a simplified einsum into a sequence of transpose, reshape, (batch_)dot or mul operations. This decomposition is converted into ONNX and executed with *onnxruntime*, version `ort-matmul`. Both versions are faster than the numpy optimized version.

# ## Another example with `bsnh,btnh=bnts`
# 
# Another case, more frequent in deep learning.

# ### Decomposition of `bsnh,btnh=bnts`

# In[17]:


seq2 = decompose_einsum_equation("bsnh,btnh->bnts", strategy='numpy', clean=True)
RenderJsDot(seq2.to_dot(size=7))


# ### ONNX version

# In[18]:


onx2 = seq2.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
get_ipython().run_line_magic('onnxview', 'onx2')


# ### Benchmark

# In[19]:


def make_model2(equation):
    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid('', 13)],
        graph=helper.make_graph(
            name='einsum_test',
            inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
            outputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, None)],
            nodes=[
                helper.make_node("Einsum", ["X", "Y"], ["A"], equation=equation)
            ]
        )
    )
    return model


def benchmark(equation, second_input_size=4):
    sess = None
    sess2 = None
    seq = None
    seq2 = None


    results = []
    for N in tqdm([2, 3, 4, 10, 20, 30, 40]):
        m1 = numpy.random.randn(10, N, N, N)
        m2 = numpy.random.randn(10 * N ** (second_input_size-1)).reshape((10, ) + (N, ) * (second_input_size-1))
        

        if seq is None:
            seq = decompose_einsum_equation(equation, clean=True)
        if seq2 is None:
            seq2 = decompose_einsum_equation(equation, clean=True, strategy='numpy')
        if sess is None:
            model = make_model2(equation)
            sess = InferenceSession(model.SerializeToString())
        if sess2 is None:
            onx = seq2.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
            sess2 = InferenceSession(onx.SerializeToString())

        res = measure_time(lambda x: numpy.einsum(equation, *x, optimize=True),
                           [m1, m2],
                           repeat=10, number=10)

        res['name'] = "numpy.einsum"
        res["N"] = N
        results.append(res)

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x),
                           [m1, m2],
                           repeat=10, number=10)
        res['name'] = "custom_einsum"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x, matmul_impl="pyf"),
                           [m1, m2],
                           repeat=10, number=10)
        res['name'] = "dec-matmul"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq2, *x, matmul_impl="pyf"),
                           [m1, m2],
                           repeat=10, number=10)
        res['name'] = "dec-batch_dot"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: sess.run(None, {'X': x[0], 'Y': x[1]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=10, number=10)
        res['name'] = "ort-einsum"
        res["N"] = N
        results.append(res)    

        res = measure_time(lambda x: sess2.run(None, {'X1': x[0], 'X2': x[1]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=10, number=10)
        res['name'] = "ort-matmul"
        res["N"] = N
        results.append(res)    
    return DataFrame(results)


df = benchmark("bsnh,btnh->bnts")
df.tail()


# In[20]:


piv = df.pivot("N", "name", "average")
piv2 = piv.copy()
np = piv["numpy.einsum"]
for c in piv2.columns:
    piv2[c] /= np
    
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
piv.plot(logy=True, logx=True, ax=ax[0])
ax[0].set_title("Benchmark einsum function\nbsnh,btnh->bnts")
piv2.plot(logy=True, logx=True, ax=ax[1])
ax[1].set_title("Benchmark einsum function\n(ratio, baseline=numpy)");


# ### Permutation
# 
# Einsum's algorithm started by aligning all matrices involved in the computation to the same dimension in the same order. But which order is the best, that's the question.

# In[21]:


equation = "bsnh,btnh->bnts"
letters = list(sorted(set([c for c in equation if "a" <= c < "z"])))
letters


# In[22]:


from itertools import permutations


def benchmark_perm(equation, number=5, second_input_size=4, repeat=3, N=15):
    
    def n_operator(seq, name):
        n = 0
        for op in seq:
            if op.name == name:
                n += 1
        return n


    def n_onnx_op(onx, name):
        n = 0
        for op in onx.graph.node:
            if op.op_type == name:
                n += 1
        return n


    def get_kind(seq):
        n = 0
        for op in seq:
            if op.name == 'batch_dot':
                return op.get_dot_kind()
        return None


    m1 = numpy.random.randn(N, N, N, N)
    m2 = numpy.random.randn(N ** second_input_size).reshape((N, ) * second_input_size)

    results = []
    for perm in tqdm(list(permutations(letters))):
        replace = {d: c for c, d in zip(letters, perm)}
        eq = equation
        for k, v in replace.items():
            eq = eq.replace(k, v.upper())
        eq = eq.lower()

        seq = decompose_einsum_equation(eq, clean=True)
        seq2 = decompose_einsum_equation(eq, clean=True, strategy='numpy')
        model = make_model2(eq)
        sess = InferenceSession(model.SerializeToString())
        onx = seq2.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
        sess2 = InferenceSession(onx.SerializeToString())

        n_tra = n_operator(seq2, 'transpose')
        n_tra_onnx = n_onnx_op(onx, 'Transpose')
        n_gemm_onnx = n_onnx_op(onx, 'Gemm')
        kind = get_kind(seq2)

        res = measure_time(lambda x: numpy.einsum(eq, *x, optimize=True),
                           [m1, m2],
                           repeat=repeat, number=number)

        res['name'] = "numpy.einsum"
        res["N"] = N
        res["eq"] = eq
        results.append(res)

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x),
                           [m1, m2],
                           repeat=repeat, number=number)
        res['name'] = "custom_einsum"
        res["N"] = N
        res["eq"] = eq
        res['transpose'] = n_tra
        res['kind'] = kind
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq, *x, matmul_impl="pyf"),
                           [m1, m2],
                           repeat=repeat, number=number)
        res['name'] = "dec-matmul"
        res["N"] = N
        res["eq"] = eq
        res['transpose'] = n_tra
        res['kind'] = kind
        results.append(res)    

        res = measure_time(lambda x: apply_einsum_sequence(seq2, *x, matmul_impl="pyf"),
                           [m1, m2],
                           repeat=repeat, number=number)
        res['name'] = "dec-batch_dot"
        res["N"] = N
        res["eq"] = eq
        res['transpose'] = n_tra
        res['kind'] = kind
        results.append(res)    

        res = measure_time(lambda x: sess.run(None, {'X': x[0], 'Y': x[1]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=repeat, number=number)
        res['name'] = "ort-einsum"
        res["N"] = N
        res["eq"] = eq
        res['transpose'] = n_tra_onnx
        res['gemm'] = n_gemm_onnx
        results.append(res)    

        res = measure_time(lambda x: sess2.run(None, {'X1': x[0], 'X2': x[1]}),
                           [m1.astype(numpy.float32), m2.astype(numpy.float32),
                            m3.astype(numpy.float32)],
                           repeat=repeat, number=number)
        res['name'] = "ort-matmul"
        res["N"] = N
        res["eq"] = eq
        res['transpose'] = n_tra_onnx
        res['gemm'] = n_gemm_onnx
        results.append(res)    
    return DataFrame(results)


df = benchmark_perm("bsnh,btnh->bnts", number=4)
df.tail()


# In[23]:


df = df.sort_values("average").reset_index(drop=True)
df.head()


# In[24]:


df.tail()


# In[25]:


piv = df.pivot("eq", "name", "average").sort_values("numpy.einsum")
    
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
piv.plot(logy=True, logx=True, ax=ax)
ax.set_title("Benchmark einsum function - bsnh,btnh->bnts");


# In[26]:


set(df['transpose'].dropna()), set(df['gemm'].dropna()), set(df['kind'].dropna())


# ## Decomposition of `bsnh,ctnh=nts`

# In[27]:


seq3 = decompose_einsum_equation("bsnh,ctnh->nts", strategy='numpy', clean=True)
RenderJsDot(seq3.to_dot(size=7))


# In[28]:


onx3 = seq3.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
get_ipython().run_line_magic('onnxview', 'onx3')


# ### Benchmark size

# In[29]:


df = benchmark("bsnh,ctnh->nts")
df.tail()


# In[30]:


piv = df.pivot("N", "name", "average")
piv2 = piv.copy()
np = piv["numpy.einsum"]
for c in piv2.columns:
    piv2[c] /= np
    
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
piv.plot(logy=True, logx=True, ax=ax[0])
ax[0].set_title("Benchmark einsum function\nbsnh,ctnh->nts")
piv2.plot(logy=True, logx=True, ax=ax[1])
ax[1].set_title("Benchmark einsum function\n(ratio, baseline=numpy)");


# ### Benchmark permutation

# In[31]:


df = benchmark_perm("bsnh,ctnh->nts", number=2, repeat=3, N=10)


# In[32]:


df = df.sort_values("average").reset_index(drop=True)
df.head()


# In[33]:


set(df['transpose'].dropna()), set(df['gemm'].dropna()), set(df['kind'].dropna())


# In[34]:


piv = df.pivot("eq", "name", "average").sort_values("numpy.einsum")
    
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
piv.plot(logy=True, logx=True, ax=ax)
ax.set_title("Benchmark einsum function");


# ### Best permutation

# One of the best permutation is `bnst,chst->shn`.

# In[35]:


seq4 = decompose_einsum_equation("bnst,chst->shn", strategy='numpy', clean=True)
RenderJsDot(seq4.to_dot(size=7))


# In[36]:


onx4 = seq4.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
get_ipython().run_line_magic('onnxview', 'onx4')


# In[37]: