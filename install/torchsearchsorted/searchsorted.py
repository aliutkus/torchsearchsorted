import torch
import numpy as np

# trying to import the CPU searchsorted
SEARCHSORTED_CPU_AVAILABLE = True
try:
    from .cpu import searchsorted_cpu_wrapper
except ImportError:
    SEARCHSORTED_CPU_AVAILABLE = False

# trying to import the CPU searchsorted
SEARCHSORTED_GPU_AVAILABLE = True
try:
    from .cuda import searchsorted_cuda_wrapper
except ImportError:
    SEARCHSORTED_GPU_AVAILABLE = False


print('CPU searchsorted available', SEARCHSORTED_CPU_AVAILABLE)
print('GPU searchosrted available', SEARCHSORTED_GPU_AVAILABLE)


def searchsorted(a, v, out=None):
    assert len(a.shape) == 2, "input `a` must be 2-D."
    assert len(v.shape) == 2, "input `v` mus(t be 2-D."
    assert (a.shape[0] == v.shape[0]
            or a.shape[0] == 1
            or v.shape[0] == 1), ("`a` and `v` must have the same number of "
                                  "rows or one of them must have only one ")
    assert a.is_cuda == v.is_cuda, ('inputs `a` and `v` must be both on '
                                    'cpu or on gpu')

    result_shape = (max(a.shape[0], v.shape[0]), v.shape[1])
    if out is not None:
        assert out.shape == result_shape, ("If the output tensor is provided, "
                                           "its shape must be correct. Here: "
                                           ''.join(result_shape))
    else:
        out = torch.zeros(*result_shape,
                          dtype=v.dtype, layout=v.layout, device=v.device)

    if a.is_cuda and not SEARCHSORTED_GPU_AVAILABLE:
        raise Exception('torchsearchsorted on CUDA device is asked, but it seems '
                        'that it is not available. Please install it')
    if not a.is_cuda and not SEARCHSORTED_CPU_AVAILABLE:
        raise Exception('torchsearchsorted on CPU is not available. '
                        'Please install it.')

    if a.is_cuda:
        searchsorted_cuda_wrapper(a, v, out)
    else:
        searchsorted_cpu_wrapper(a, v, out)

    return out
