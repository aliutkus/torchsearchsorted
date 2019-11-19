import numpy as np


def numpy_searchsorted(a: np.ndarray, v: np.ndarray,
                       out: np.ndarray=None, side='left') -> np.ndarray:
    """Batch-wise version of numpy's searchsorted"""
    a = np.asarray(a)
    v = np.asarray(v)
    a, v = broadcast_arrays(a, v, axis=0)
    if out is None:
        out = np.empty(v.shape, dtype=np.long)
    for i in range(v.shape[0]):
        out[i] = np.searchsorted(a[i], v[i], side=side)
    return out


def broadcast_arrays(*arrays, axis=0):
    """Broadcast arrays along one axis, leaving other axes unchanged"""
    if axis < 0:
        raise ValueError(f"Negative axis not supported, got {axis}")
    axis_size = max(a.shape[axis] for a in arrays)
    return [
        np.broadcast_to(a, (*a.shape[:axis], axis_size, *a.shape[axis + 1:]))
        for a in arrays
    ]
