import torch
import cusearchsorted


def searchsorted(a, v, out=None):
    assert len(a.shape) == 2, "input `a` must be 2-D."
    assert len(v.shape) == 2, "input `v` mus(t be 2-D."
    assert (a.shape[0] == v.shape[0]
            or a.shape[1] == 1
            or v.shape[1] == 1), ("`a` and `v` must have the same number of "
                                  "rows or one of them must have only one ")

    if out is not None:
        assert out.shape == v.shape, ("If the output tensor is provided, its "
                                      "shape must match that of `v`.")
    else:
        out = torch.zeros_like(v)

    cusearchsorted.searchsorted_cuda_wrapper(a, v, out)
    return out
