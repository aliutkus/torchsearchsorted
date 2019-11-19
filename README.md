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
After setting up an environment with pytorch >= 1.3, run either of these commands from the root folder of the repo: 

```bash
pip install -v .
```

```bash
python setup.py install -v
```

The verbose flag `-v` is not mandatory, but it will print whether the installer was able to find `nvcc` and install the CUDA version of `torchsearchsorted`.
If you're having problems with the installation, make sure `nvcc` and `gcc` are installed and available in your path, e.g.:
```bash
export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH}"

which gcc
which nvcc

pip install -v .
```

## Usage

```python
import torch
from torchsearchsorted import searchsorted

a = torch.sort(torch.randn(5000, 300, device='cuda'), dim=1)[0]
v = torch.randn(5000, 100, device='cuda')
out = searchsorted(a, v)
```


## Testing and benchmarking

Install test dependencies and run the unit tests:
```bash
pip install '.[test]'
pytest -v
```

Run [benchmark.py](examples/benchmark.py) for a speed comparison: 
```bash
python examples/benchmark.py
```
```text
Benchmark searchsorted:
- a [5000 x 300]
- v [5000 x 100]
- reporting fastest time of 20 runs
- each run executes searchsorted 100 times

Numpy: 	3.4524286500000017
CPU: 	10.617608329001087
CUDA: 	0.00124932999824523
```
