#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # ONNX and FFT
# 
# ONNX does not fully support complex yet. It does not have any FFT operators either. What if we need them anyway?

# In[1]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[2]:


get_ipython().run_line_magic('load_ext', 'mlprodict')


# In[3]:


import numpy
numpy.__version__


# ## Python implementation of RFFT
# 
# We try to replicate [numpy.rfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html).

# In[4]:


import numpy


def almost_equal(a, b, error=1e-5):
    """
    The function compares two matrices, one may be complex. In that case,
    this matrix is changed into a new matrix with a new first dimension,
    [0,::] means real part, [1,::] means imaginary part.
    """
    if a.dtype in (numpy.complex64, numpy.complex128):
        dtype = numpy.float64 if a.dtype == numpy.complex128 else numpy.float32
        new_a = numpy.empty((2,) + a.shape).astype(dtype)
        new_a[0] = numpy.real(a)
        new_a[1] = numpy.imag(a)
        return almost_equal(new_a, b, error)
    if b.dtype in (numpy.complex64, numpy.complex128):
        return almost_equal(b, a, error)
    if a.shape != b.shape:
        raise AssertionError("Shape mismatch %r != %r." % (a.shape, b.shape))
    diff = numpy.abs(a.ravel() - b.ravel()).max()
    if diff > error:
        raise AssertionError("Mismatch max diff=%r > %r." % (diff, error))


def dft_real_cst(N, fft_length):
    n = numpy.arange(N)
    k = n.reshape((N, 1)).astype(numpy.float64)
    M = numpy.exp(-2j * numpy.pi * k * n / fft_length)
    both = numpy.empty((2,) + M.shape)
    both[0, :, :] = numpy.real(M)
    both[1, :, :] = numpy.imag(M)
    return both


def dft_real(x, fft_length=None, transpose=True):
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
        N = 1
    else:
        N = x.shape[0]        
    C = x.shape[-1] if transpose else x.shape[-2]
    if fft_length is None:
        fft_length = x.shape[-1]
    size = fft_length // 2 + 1

    cst = dft_real_cst(C, fft_length)
    if transpose:
        x = numpy.transpose(x, (1, 0))
        a = cst[:, :, :fft_length]
        b = x[:fft_length]
        res = numpy.matmul(a, b)
        res = res[:, :size, :]
        return numpy.transpose(res, (0, 2, 1))
    else:
        a = cst[:, :, :fft_length]
        b = x[:fft_length]
        return numpy.matmul(a, b)


rnd = numpy.random.randn(5, 7).astype(numpy.float32)
fft_np = numpy.fft.rfft(rnd)
fft_cus = dft_real(rnd)
fft_np


# Function `almost_equal` verifies both functions return the same results.

# In[5]:


almost_equal(fft_np, fft_cus)


# Let's do the same with `fft_length < shape[1]`.

# In[6]:


fft_np3 = numpy.fft.rfft(rnd, n=3)
fft_cus3 = dft_real(rnd, fft_length=3)
fft_np3


# In[7]:


almost_equal(fft_np3, fft_cus3)


# ## RFFT in ONNX
# 
# Let's assume first the number of column of the input matrix is fixed. The result of function `dft_real_cst` can be considered as constant.

# In[8]:


from typing import Any
import mlprodict.npy.numpy_onnx_impl as npnx
from mlprodict.npy import onnxnumpy_np
from mlprodict.npy.onnx_numpy_annotation import NDArrayType
# from mlprodict.onnxrt import OnnxInference

@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=('T',)))
def onnx_rfft(x, fft_length=None):
    if fft_length is None:
        raise RuntimeError("fft_length must be specified.")
    
    size = fft_length // 2 + 1
    cst = dft_real_cst(fft_length, fft_length).astype(numpy.float32)
    xt = npnx.transpose(x, (1, 0))
    res = npnx.matmul(cst[:, :, :fft_length], xt[:fft_length])[:, :size, :]
    return npnx.transpose(res, (0, 2, 1))

fft_onx = onnx_rfft(rnd, fft_length=rnd.shape[1])
fft_onx


# In[9]:


almost_equal(fft_cus, fft_onx)


# The corresponding ONNX graph is the following:

# In[10]:


get_ipython().run_line_magic('onnxview', 'onnx_rfft.to_onnx()')


# In[11]:


fft_onx3 = onnx_rfft(rnd, fft_length=3)
almost_equal(fft_cus3, fft_onx3)


# ## FFT 2D
# 
# Below the code for complex features.

# In[12]:


def _DFT_cst(N, fft_length, trunc=True):
    n = numpy.arange(N)
    k = n.reshape((N, 1)).astype(numpy.float64)
    M = numpy.exp(-2j * numpy.pi * k * n / fft_length)
    return M[:fft_length // 2 + 1] if trunc else M

def DFT(x, fft_length=None, axis=1):
    if axis == 1:
        x = x.T
    if fft_length is None:
        fft_length = x.shape[0]
    cst = _DFT_cst(x.shape[0], fft_length, trunc=axis==1)
    if axis == 1:
        return numpy.matmul(cst, x).T
    return numpy.matmul(cst, x)

def fft2d_(mat, fft_length):
    mat = mat[:fft_length[0], :fft_length[1]]
    res = mat.copy()
    res = DFT(res, fft_length[1], axis=1)
    res = DFT(res, fft_length[0], axis=0)
    return res[:fft_length[0], :fft_length[1]//2 + 1]


rnd = numpy.random.randn(5, 7).astype(numpy.float32)
fft2d_np_ = fft2d_(rnd, rnd.shape)
fft2d_np = numpy.fft.rfft2(rnd)
fft2d_np_


# In[13]:


almost_equal(fft2d_np_, fft2d_np)


# It implies the computation of two FFT 1D along both axes. However, as ONNX does not support complex, it needs to be rewritten with only real numbers. The algorithm can be summarized into this formula $FFT(FFT(x, axis=1), axis=0)$. If *x* is real, $FFT(x, .)$ is complex. We still assume *x* is real, it then becomes (FFT is a linear operator, so $FFT(ix)=i FFT(x)$):
# 
# * $y = FFT(x, axis=1)$
# * $z_r = FFT(Real(y), axis=0)$, $z_i = FFT(Imag(y), axis=0)$
# * $z = z_r + i z_i$
# 
# *z* is the desired output. The following implementation is probably not the most efficient one. It avoids inplace computation as ONNX does like that.

# In[14]:


def fft2d(mat, fft_length):
    mat = mat[:fft_length[0], :fft_length[1]]
    res = mat.copy()
    
    # first FFT
    res = dft_real(res, fft_length=fft_length[1], transpose=True)
    
    # second FFT decomposed on FFT on real part and imaginary part
    res2_real = dft_real(res[0], fft_length=fft_length[0], transpose=False)
    res2_imag = dft_real(res[1], fft_length=fft_length[0], transpose=False)    
    res2_imag2 = numpy.vstack([-res2_imag[1:2], res2_imag[:1]])
    res = res2_real + res2_imag2
    size = fft_length[1]//2 + 1
    return res[:, :fft_length[0], :size]


fft2d_np = numpy.fft.rfft2(rnd)
fft2d_cus = fft2d(rnd, rnd.shape)
almost_equal(fft2d_np, fft2d_cus)


# In[15]:


fft2d_np


# In[16]:


fft2d_cus


# And with a different `fft_length`.

# In[17]:


fft2d_np = numpy.fft.rfft2(rnd, (4, 6))
fft2d_cus = fft2d(rnd, (4, 6))
almost_equal(fft2d_np[:4, :], fft2d_cus)


# ## FFT 2D in ONNX
# 
# We use again the numpy API for ONNX.

# In[18]:


def onnx_rfft_1d(x, fft_length=None, transpose=True):
    if fft_length is None:
        raise RuntimeError("fft_length must be specified.")
    
    size = fft_length // 2 + 1
    cst = dft_real_cst(fft_length, fft_length).astype(numpy.float32)
    if transpose:
        xt = npnx.transpose(x, (1, 0))
        res = npnx.matmul(cst[:, :, :fft_length], xt[:fft_length])[:, :size, :]
        return npnx.transpose(res, (0, 2, 1))
    else:
        return npnx.matmul(cst[:, :, :fft_length], x[:fft_length])


@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=('T',)))
def onnx_rfft_2d(x, fft_length=None):
    mat = x[:fft_length[0], :fft_length[1]]
    
    # first FFT
    res = onnx_rfft_1d(mat, fft_length=fft_length[1], transpose=True)
    
    # second FFT decomposed on FFT on real part and imaginary part
    res2_real = onnx_rfft_1d(res[0], fft_length=fft_length[0], transpose=False)
    res2_imag = onnx_rfft_1d(res[1], fft_length=fft_length[0], transpose=False)    
    res2_imag2 = npnx.vstack(-res2_imag[1:2], res2_imag[:1])
    res = res2_real + res2_imag2
    size = fft_length[1]//2 + 1
    return res[:, :fft_length[0], :size]


fft2d_cus = fft2d(rnd, rnd.shape)
fft2d_onx = onnx_rfft_2d(rnd, fft_length=rnd.shape)
almost_equal(fft2d_cus, fft2d_onx)


# The corresponding ONNX graph.

# In[19]:


get_ipython().run_line_magic('onnxview', 'onnx_rfft_2d.to_onnx()')


# In[20]:


with open("fft2d.onnx", "wb") as f:
    f.write(onnx_rfft_2d.to_onnx().SerializeToString())


# With a different `fft_length`.

# In[21]:


fft2d_cus = fft2d(rnd, (4, 5))
fft2d_onx = onnx_rfft_2d(rnd, fft_length=(4, 5))
almost_equal(fft2d_cus, fft2d_onx)


# This implementation of FFT in ONNX assumes shapes and fft lengths are constant. Otherwise, the matrix returned by function `dft_real_cst` must be converted as well. That's left as an exercise.

# ## FFT2D with shape (3,1,4)
# 
# Previous implementation expects the input matrix to have two dimensions. It fails with 3.

# In[22]:


shape = (3, 1, 4)
fft_length = (1, 4)
rnd = numpy.random.randn(*list(shape)).astype(numpy.float32)
fft2d_numpy = numpy.fft.fft2(rnd, fft_length)
fft2d_numpy.shape


# In[23]:


fft2d_numpy


# In[24]:


try:
    fft2d_cus = fft2d(rnd, fft_length)
except Exception as e:
    print(e)
# fft2d_onx = onnx_rfft_2d(rnd, fft_length=fft_length)


# ### numpy version
# 
# Let's do it again with numpy first. [fft2](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html) performs `fft2` on the last two axis as many times as the first axis. The goal is still to have an implementation which works for any dimension.

# In[25]:


conc = []
for i in range(rnd.shape[0]):
    f2 = fft2d(rnd[i], fft_length)
    conc.append(numpy.expand_dims(f2, 0))
res = numpy.vstack(conc).transpose(1, 0, 2, 3)
almost_equal(fft2d_numpy[:, :, :3], res)


# It works. And now a more efficient implementation. It is better to read [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) description before. To summarize, a third axis is equivalent to many matrix multiplications over the last two axes, as many as the dimension of the first axis: ``matmul(A[I,J,K], B[I,K,L]) --> C[I,J,L]``. Broadcasting also works... ``matmul(A[1,J,K], B[I,K,L]) --> C[I,J,L]``.

# In[26]:


def dft_real_d3(x, fft_length=None, transpose=True):
    if len(x.shape) != 3:
        raise RuntimeError("Not implemented for shape=%r." % x.shape)
    N = x.shape[1]
    C = x.shape[-1] if transpose else x.shape[-2]
    if fft_length is None:
        fft_length = x.shape[-1]
    size = fft_length // 2 + 1

    cst = dft_real_cst(C, fft_length)
    if transpose:
        x = numpy.transpose(x, (0, 2, 1))
        a = cst[:, :, :fft_length]
        b = x[:, :fft_length, :]
        a = numpy.expand_dims(a, 0)
        b = numpy.expand_dims(b, 1)
        res = numpy.matmul(a, b)
        res = res[:, :, :size, :]
        return numpy.transpose(res, (1, 0, 3, 2))
    else:
        a = cst[:, :, :fft_length]
        b = x[:, :fft_length, :]
        a = numpy.expand_dims(a, 0)
        b = numpy.expand_dims(b, 1)
        res = numpy.matmul(a, b)
        return numpy.transpose(res, (1, 0, 2, 3))


def fft2d_d3(mat, fft_length):
    mat = mat[:, :fft_length[-2], :fft_length[-1]]
    res = mat.copy()
    
    # first FFT
    res = dft_real_d3(res, fft_length=fft_length[-1], transpose=True)
    
    # second FFT decomposed on FFT on real part and imaginary part
    res2_real = dft_real_d3(res[0], fft_length=fft_length[-2], transpose=False)
    res2_imag = dft_real_d3(res[1], fft_length=fft_length[-2], transpose=False)
    res2_imag2 = numpy.vstack([-res2_imag[1:2], res2_imag[:1]])
    res = res2_real + res2_imag2
    size = fft_length[-1]//2 + 1
    return res[:, :, :fft_length[-2], :size]


def fft2d_any(mat, fft_length):
    new_shape = (-1, ) + mat.shape[-2:]
    mat2 = mat.reshape(new_shape)
    f2 = fft2d_d3(mat2, fft_length)
    new_shape = (2, ) + mat.shape[:-2] + f2.shape[-2:]
    return f2.reshape(new_shape)


shape = (3, 1, 4)
fft_length = (1, 4)
rnd = numpy.random.randn(*list(shape)).astype(numpy.float32)
fft2d_numpy = numpy.fft.fft2(rnd, fft_length)
fft2d_cus = fft2d_any(rnd, fft_length)
almost_equal(fft2d_numpy[..., :3], fft2d_cus)


# We check with more shapes to see if the implementation works for all of them.

# In[27]:


for shape in [(3, 1, 4), (5, 7), (3, 5, 7), (7, 5)]:
    for fft_length in [shape[-2:], (1, shape[-1]),
                       (min(2, shape[-2]), shape[-1]),
                       (shape[-2], 2),
                       (min(3, shape[-2]), min(4, shape[-2]))]:
        x  = numpy.random.randn(*list(shape)).astype(numpy.float32)
        fnp = numpy.fft.fft2(x, fft_length)
        if len(fnp.shape) == 2:
            fn= numpy.expand_dims(fnp, 0)
        try:
            cus = fft2d_any(x, fft_length)
        except IndexError as e:
            print("ERR x.shape=%r length=%r error=%r" % (x.shape, fft_length, e))
            continue
        try:
            almost_equal(fnp[..., :cus.shape[-1]], cus)
        except (AssertionError, IndexError) as e:
            print("DIS x.shape=%r length=%r error=%r  output shape=%r or %r" % (
                x.shape, fft_length, e, fnp.shape, cus.shape))
            continue
        print("OK  x.shape=%r length=%r output shape=%r or %r" % (
            x.shape, fft_length, fnp.shape, cus.shape))


# ### ONNX version
# 
# Let's look into the differences first.

# In[28]:


get_ipython().run_line_magic('load_ext', 'pyquickhelper')


# In[29]:


get_ipython().run_cell_magic('html', '', '<style>\ntable td, table th, table tr {text-align:left !important; white-space: pre;}\n</style>')


# In[30]:


import inspect
text1 = inspect.getsource(dft_real)
text2 = inspect.getsource(dft_real_d3)
get_ipython().run_line_magic('codediff', 'text1 text2 --verbose 1 --two 1')


# In[31]:


text1 = inspect.getsource(fft2d)
text2 = inspect.getsource(fft2d_d3)
get_ipython().run_line_magic('codediff', 'text1 text2 --verbose 1 --two 1')


# In[32]:


def onnx_rfft_3d_1d(x, fft_length=None, transpose=True):
    if fft_length is None:
        raise RuntimeError("fft_length must be specified.")
    
    size = fft_length // 2 + 1
    cst = dft_real_cst(fft_length, fft_length).astype(numpy.float32)
    if transpose:
        xt = npnx.transpose(x, (0, 2, 1))
        a = cst[:, :, :fft_length]
        b = xt[:, :fft_length, :]
        a = npnx.expand_dims(a, 0)
        b = npnx.expand_dims(b, 1)
        res = npnx.matmul(a, b)
        res2 = res[:, :size, :]
        return npnx.transpose(res2, (1, 0, 3, 2))
    else:
        a = cst[:, :, :fft_length]
        b = x[:, :fft_length, :]
        a = npnx.expand_dims(a, 0)
        b = npnx.expand_dims(b, 1)
        res = npnx.matmul(a, b)
        return npnx.transpose(res, (1, 0, 2, 3))      
    

def onnx_rfft_3d_2d(x, fft_length=None):
    mat = x[:, :fft_length[-2], :fft_length[-1]]
    
    # first FFT
    res = onnx_rfft_3d_1d(mat, fft_length=fft_length[-1], transpose=True)
    
    # second FFT decomposed on FFT on real part and imaginary part
    res2_real = onnx_rfft_3d_1d(res[0], fft_length=fft_length[0], transpose=False)
    res2_imag = onnx_rfft_3d_1d(res[1], fft_length=fft_length[0], transpose=False)    
    res2_imag2 = npnx.vstack(-res2_imag[1:2], res2_imag[:1])
    res = res2_real + res2_imag2
    size = fft_length[1]//2 + 1
    return res[:, :, :fft_length[-2], :size]


@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=('T',)))
def onnx_rfft_2d_any(x, fft_length=None):
    new_shape = npnx.concat(
        numpy.array([-1], dtype=numpy.int64), x.shape[-2:], axis=0)
    mat2 = x.reshape(new_shape)
    f2 = onnx_rfft_3d_2d(mat2, fft_length)
    new_shape = npnx.concat(
        numpy.array([2], dtype=numpy.int64), x.shape[:-2], f2.shape[-2:])
    return f2.reshape(new_shape)


shape = (3, 1, 4)
fft_length = (1, 4)
rnd = numpy.random.randn(*list(shape)).astype(numpy.float32)
fft2d_cus = fft2d_any(rnd, fft_length)
fft2d_onx = onnx_rfft_2d_any(rnd, fft_length=fft_length)
almost_equal(fft2d_cus, fft2d_onx)


# Let's do the same comparison.

# In[33]:


for shape in [(3, 1, 4), (5, 7), (3, 5, 7), (7, 5)]:
    for fft_length in [shape[-2:], (1, shape[-1]),
                       (min(2, shape[-2]), shape[-1]),
                       (shape[-2], 2),
                       (min(3, shape[-2]), min(4, shape[-2]))]:
        x  = numpy.random.randn(*list(shape)).astype(numpy.float32)
        if len(fnp.shape) == 2:
            fn= numpy.expand_dims(fnp, 0)
        try:
            cus = fft2d_any(x, fft_length)
        except IndexError as e:
            print("ERR x.shape=%r length=%r error=%r" % (x.shape, fft_length, e))
            continue
        try:
            onx = onnx_rfft_2d_any(x, fft_length=fft_length)
        except IndexError as e:
            print("ERR x.shape=%r length=%r error=%r" % (x.shape, fft_length, e))
            continue
        try:
            almost_equal(onx, cus)
        except (AssertionError, IndexError) as e:
            print("DIS x.shape=%r length=%r error=%r  output shape=%r or %r" % (
                x.shape, fft_length, e, fnp.shape, cus.shape))
            continue
        print("OK  x.shape=%r length=%r output shape=%r or %r" % (
            x.shape, fft_length, fnp.shape, cus.shape))


# There is one issue with ``fft_length=(1, 1)`` but that case is out of scope.

# ### ONNX graph

# In[34]:


key = list(onnx_rfft_2d_any.signed_compiled)[0]
get_ipython().run_line_magic('onnxview', 'onnx_rfft_2d_any.signed_compiled[key].compiled.onnx_')


# In[35]:


with open("fft2d_any.onnx", "wb") as f:
    key = list(onnx_rfft_2d_any.signed_compiled)[0]
    f.write(onnx_rfft_2d_any.signed_compiled[key].compiled.onnx_.SerializeToString())


# Let's check the intermediate results.

# In[36]:


key = list(onnx_rfft_2d_any.signed_compiled)[0]
key


# In[37]:


from mlprodict.onnxrt import OnnxInference

x = numpy.random.randn(3, 1, 4).astype(numpy.float32)
onx = onnx_rfft_2d_any.signed_compiled[key].compiled.onnx_
oinf = OnnxInference(onx)
oinf.run({'x': x}, verbose=1, fLOG=print)


# In[38]: