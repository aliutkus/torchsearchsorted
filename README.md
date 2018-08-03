# Pytorch Custom CUDA kernel for searchsorted

This repository is an implementation of the searchsorted function to work for pytorch CUDA Tensors.

It is derived from the awesome [Pytorch Custom CUDA kernel Tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial)

**Disclaimer**

```
This function has not been heavily tested. Use at your own risks
```
## Description

Implements a function `searchsorted(a, v)` that works just like the [numpy version](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) except that `a` and `v` are matrices.
* `a` is of shape either `(1, ncols_a)` or `(nrows, ncols_a)`
* `b` is of shape either `(1, ncols_v)` or `(nrows, ncols_v)`.

the output is of size as `(nrows, ncols_v)`. Only works with pytorch tensors that are
already on the GPU.

## Installation

Just `make`


## Testing

Try `python test.py` with `torch` available. Tested on Pytorch v0.4.


```
Searching for 50000x1000 values in 50000x300 entries
GPU:  searchsorted in 119.483ms
CPU:  searchsorted in 8625.762ms
    difference: 0.0
```
