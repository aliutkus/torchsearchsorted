import warnings
from typing import Optional

import torch

from torchsearchsorted.cpu import searchsorted_cpu_wrapper

if torch.cuda.is_available():
    try:
        from torchsearchsorted.cuda import searchsorted_cuda_wrapper
    except ImportError as e:
        warnings.warn("PyTorch is installed with CUDA support, but "
                      "torchsearchsorted for CUDA was not installed, "
                      "please repeat the installation or avoid passing "
                      "CUDA tensors to the `searchsorted`.")


def searchsorted(a: torch.Tensor,
                 v: torch.Tensor,
                 out: Optional[torch.LongTensor] = None,
                 side='left') -> torch.LongTensor:
    if a.ndimension() != 2:
        raise ValueError(f"Input `a` must be 2D, got shape {a.shape}")
    if v.ndimension() != 2:
        raise ValueError(f"Input `v` must be 2D, got shape {v.shape}")
    if a.device != v.device:
        raise ValueError(f"Inputs `a` and `v` must on the same device, "
                         f"got {a.device} and {v.device}")

    a, v = broadcast_tensors(a, v, dim=0)

    if out is not None:
        if out.shape != v.shape:
            raise ValueError(f"Output `out` must have the same shape as `v`, "
                             f"got {out.shape} and {a.shape}")
        if out.device != v.device:
            raise ValueError(f"Output `out` must be on the same device as `v`"
                             f"device, got {out.device} and {v.device}")
        if out.dtype != torch.long:
            raise ValueError(f"Output `out` must have dtype `torch.long`, "
                             f"got {out.dtype}")
    else:
        out = torch.empty(v.shape, device=v.device, dtype=torch.long)

    left_side = side == 'left'
    if a.is_cuda:
        searchsorted_cuda_wrapper(a, v, out, left_side)
    else:
        searchsorted_cpu_wrapper(a, v, out, left_side)

    return out


def broadcast_tensors(*tensors, dim=0):
    """Broadcast tensors along one dimension, leaving others dims unchanged"""
    if dim < 0:
        raise ValueError(f"Negative dimensions not supported, got {dim}")
    dim_size = max(t.shape[dim] for t in tensors)
    return [t.expand(*t.shape[:dim], dim_size, *t.shape[dim + 1:])
            for t in tensors]
