import numpy as np


def numpy_searchsorted(a: np.ndarray, v: np.ndarray, side='left'):
    """Numpy version of searchsorted that works batch-wise on pytorch tensors
    """
    out = np.empty_like(v, dtype=np.long)

    if a.ndim == 1:
        out[:] = np.searchsorted(a, v, side=side)
    elif a.ndim == 2:
        for row_idx in range(a.shape[0]):
            out[row_idx, :] = np.searchsorted(a[row_idx], v[row_idx], side=side)
    return out
