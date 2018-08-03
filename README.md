# Pytorch Custom CUDA kernel for searchsorted

This repository is an implementation of the searchsorted function to work for pytorch CUDA Tensors.

It is derived from the awesome [Pytorch Custom CUDA kernel Tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial)

## Description

Implements a function `searchsorted(a, v)` that works just like the [numpy version](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) except that `a` and `v` are assumed to be matrices of respective shape `(nrows, ncols_a)` and `(nrows, ncols_v)`.

the output is of the same size as `v`. Only works with pytorch tensors that are
already on the GPU.


**Disclaimers**

* This function has not been heavily tested. Use at your own risks
* When `a` is not sorted, the results vary from numpy's version. But I decided not to care about this because the function should not be called in this case.


## Installation

Just `make`


## Testing

Try `python test.py` with `torch` available. Tested on Pytorch v0.4.


```
Searching 50000x1000 values in 50000x300 entries
 GPU:  searchsorted in 143.391ms
 CPU:  searchsorted in 8384.157ms
     difference: 0.0
```
