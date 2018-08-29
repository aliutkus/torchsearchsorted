# Pytorch Custom CUDA kernel for searchsorted

This repository is an implementation of the searchsorted function to work for pytorch CUDA Tensors.

It is derived from the awesome [Pytorch Custom CUDA kernel Tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial)

## Description

Implements a function `searchsorted(a, v)` that works just like the [numpy version](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) except that `a` and `v` are matrices.
* `a` is of shape either `(1, ncols_a)` or `(nrows, ncols_a)`
* `b` is of shape either `(1, ncols_v)` or `(nrows, ncols_v)`.

the output is of size as `(nrows, ncols_v)`. Only works with pytorch tensors that are
already on the GPU.


**Disclaimers**

* This function has not been heavily tested. Use at your own risks
* When `a` is not sorted, the results vary from numpy's version. But I decided not to care about this because the function should not be called in this case.


## Installation

Just `make`. This will compile and install the CUDA searchsorted module into the
`searchsorted` sub-directory.

## Usage

With the `searchsorted` directory somewhere in the Python PATH, just do `import searchsorted`. For instance, I typically clone this repo in my code, and then:

```
from pytorch_searchsorted.searchsorted import searchsorted
```


## Testing

Try `python test.py` with `torch` available for an example. Tested on Pytorch v0.4.1


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
