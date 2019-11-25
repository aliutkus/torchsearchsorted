# Pytorch Custom CUDA kernel for searchsorted

This repository is an implementation of the searchsorted function to work for pytorch CUDA Tensors. Initially derived from the great [C extension tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial), but totally changed since then because building C extensions is not available anymore on pytorch 1.0.


> Warning: only works with pytorch > v1.3 and CUDA >= v10.1

## Description

Implements a function `searchsorted(a, v, out, side)` that works just like the [numpy version](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) except that `a` and `v` are matrices.
* `a` is of shape either `(1, ncols_a)` or `(nrows, ncols_a)`
* `b` is of shape either `(1, ncols_v)` or `(nrows, ncols_v)`.
* `out` is either `None` or of shape `(nrows, ncols_v)`. If provided and of the right shape, the result is put there. This is to avoid costly memory allocations if the user already did it.
* `side` is either "left" or "right". See the [numpy doc](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted).

the output is of size as `(nrows, ncols_v)`. If all input tensors are on GPU, a cuda version will be called. Otherwise, it will be on CPU.


**Disclaimers**

* This function has not been heavily tested. Use at your own risks
* When `a` is not sorted, the results vary from numpy's version. But I decided not to care about this because the function should not be called in this case.
* In some cases, the results vary from numpy's version. However, as far as I could see, this only happens when values are equal, which means we actually don't care about the order in which this value is added. I decided not to care about this also.


## Installation

Just `python setup.py install`, in the root folder of this repo. This will compile
and install the torchsearchsorted module.
be careful that sometimes, `nvcc` needs versions of `gcc` and `g++` that are older than those found by default on the system. If so, just create symbolic links to the right versions in your cuda/bin folder (where `nvcc` is)

For instance, on my machine, I had `gcc` and `g++` v9 installed, but `nvcc` required v8.
So I had to do:

> sudo apt-get install g++-8 gcc-8  
> sudo ln -s /usr/bin/gcc-8 /usr/local/cuda-10.1/bin/gcc  
> sudo ln -s /usr/bin/g++-8 /usr/local/cuda-10.1/bin/g++  

be careful that you need pytorch to be installed on your system. The code was tested on pytorch v1.3

## Usage

Just import the torchsearchsorted package after installation. I typically do:

```
from torchsearchsorted import searchsorted
```


## Testing

Try `python test.py` with `torch` available for an example.

```
Looking for 50000x1000 values in 50000x300 entries
NUMPY:  searchsorted in 4851.592ms
CPU:  searchsorted in 4805.432ms
    difference between CPU and NUMPY: 0.000
GPU:  searchsorted in 1.055ms
    difference between GPU and NUMPY: 0.000

Looking for 50000x1000 values in 50000x300 entries
NUMPY:  searchsorted in 4333.964ms
CPU:  searchsorted in 4753.958ms
    difference between CPU and NUMPY: 0.000
GPU:  searchsorted in 0.391ms
    difference between GPU and NUMPY: 0.000
```
The first run comprises the time of allocation, while the second one does not.
