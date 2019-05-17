# Pytorch Custom CUDA kernel for searchsorted

This repository is an implementation of the searchsorted function to work for pytorch CUDA Tensors. Initially derived from the great [C extension tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial), but totally changed since then because building C extensions is not available anymore on pytorch 1.0.


> Warning: only works with pytorch > v1.0 and CUDA > v10

## Description

Implements a function `searchsorted(a, v, out)` that works just like the [numpy version](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) except that `a` and `v` are matrices.
* `a` is of shape either `(1, ncols_a)` or `(nrows, ncols_a)`
* `b` is of shape either `(1, ncols_v)` or `(nrows, ncols_v)`.
* `out` is either `None` or of shape `(nrows, ncols_v)`. If provided and of the right shape, the result is put there. This is to avoid costly memory allocations if the user already did it.

the output is of size as `(nrows, ncols_v)`. If all input tensors are on GPU, a cuda version will be called. Otherwise, it will be on CPU.


**Disclaimers**

* This function has not been heavily tested. Use at your own risks
* When `a` is not sorted, the results vary from numpy's version. But I decided not to care about this because the function should not be called in this case.


## Installation

Just `python setup.py install`, in the root folder of this repo. This will compile
and install the torchsearchsorted module.
be careful that sometimes, `nvcc` needs versions of `gcc` and `g++` that are older than those found by default on the system. If so, just create symbolic links to the right versions in your cuda/bin folder (where `nvcc` is)

be careful that you need pytorch to be installed on your system. The code was tested on pytorch v1.0.1

## Usage

Just import the torchsearchsorted package after installation. I typically do:

```
from torchsearchsorted import searchsorted
```


## Testing

Try `python test.py` with `torch` available for an example.

```
Searching for 50000x1000 values in 50000x300 entries
GPU:  searchsorted in 119.483ms
CPU:  searchsorted in 8625.762ms
    difference: 0.0
Looking for 50000x1000 values in 50000x300 entries
GPU:  searchsorted in 0.142ms
CPU:  searchsorted in 7337.219ms
    difference: 0.0
```
The first run comprises the time of allocation, while the second one does not.
