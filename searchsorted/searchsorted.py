import torch
from .cusearchsorted import searchsorted_cuda_wrapper


def searchsorted(a, v, out=None):
    """Implements a function `searchsorted(a, v)` that works just like the
    numpy version except that `a` and `v` are matrices.
    * `a` is of shape either `(1, ncols_a)` or `(nrows, ncols_a)`
    * `b` is of shape either `(1, ncols_v)` or `(nrows, ncols_v)`.
    * `out` is either `None` or of shape `(nrows, ncols_v)`. If provided and
    of the right shape, the result is put there. This is to avoid costly memory
    allocations if the user already did it.
    
    the output is of size as `(nrows, ncols_v)`. Only works with pytorch
    tensors that are already on the GPU.

    """
    assert a.is_cuda and v.is_cuda, "Input tensors must all be on the GPU"
    assert len(a.shape) == 2, "input `a` must be 2-D."
    assert len(v.shape) == 2, "input `v` mus(t be 2-D."
    assert (a.shape[0] == v.shape[0]
            or a.shape[0] == 1
            or v.shape[0] == 1), ("`a` and `v` must have the same number of "
                                  "rows or one of them must have only one ")

    result_shape = (max(a.shape[0], v.shape[0]), v.shape[1])
    if out is not None:
        assert out.shape == result_shape, ("If the output tensor is provided, "
                                           "its shape must be correct. Here: "
                                           ''.join(result_shape))
    else:
        out = torch.zeros(*result_shape,
                          dtype=v.dtype, layout=v.layout, device=v.device)

    searchsorted_cuda_wrapper(a, v, out)
    return out
