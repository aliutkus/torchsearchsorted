import pytest

import torch
import numpy as np
from torchsearchsorted import searchsorted

from utils import numpy_searchsorted


def test_searchsorted_output_dtype(device):
    B = 100
    A = 50
    V = 12

    a = torch.sort(torch.rand(B, V, device=device), dim=1)[0]
    v = torch.rand(B, A, device=device)

    out = searchsorted(a, v)
    out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy())
    assert out.dtype == torch.long
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)

    out = torch.empty(v.shape, dtype=torch.long, device=device)
    searchsorted(a, v, out)
    assert out.dtype == torch.long
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)


@pytest.mark.parametrize('B,A,V', [(100, 50, 12), (1000, 5000, 1200)])
def test_searchsorted_correct(B, A, V, device):
    a = torch.sort(torch.rand(B, V, device=device), dim=1)[0]
    v = torch.rand(B, A, device=device)

    out = searchsorted(a, v)
    out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy())
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)
