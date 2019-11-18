import pytest

import torch
import numpy as np
from torchsearchsorted import searchsorted, numpy_searchsorted
from itertools import product, repeat


def test_output_dtype():
    B = 100
    A = 50
    V = 12

    a = torch.sort(torch.rand(B, A), dim=1)[0]
    v = torch.rand(B, V)

    out = searchsorted(a, v)
    assert out.dtype == torch.long

    out = torch.empty(v.shape, dtype=torch.long)
    searchsorted(a, v, out)
    assert out.dtype == torch.long

    with pytest.raises(ValueError):
        out = torch.empty(v.shape, dtype=torch.float)
        searchsorted(a, v, out)


def test_broadcast_batch_dim():
    # Batch dimension:
    # (B, A), (B, V) -> (B, A), (B, V)
    # (B, A), (1, V) -> (B, A), (B, V)
    # (1, A), (B, V) -> (B, A), (B, V)
    # (1, A), (1, V) -> (1, A), (1, V)
    # (X, A), (Y, V) -> RuntimeError

    B = 6
    A = 3
    V = 4

    a = torch.sort(torch.rand(B, A), dim=1)[0]
    v = torch.rand(B, V)
    out = searchsorted(a, v)
    assert out.shape == (B, V)

    a = torch.sort(torch.rand(1, A), dim=1)[0]
    v = torch.rand(B, V)
    out = searchsorted(a, v)
    assert out.shape == (B, V)

    a = torch.sort(torch.rand(B, A), dim=1)[0]
    v = torch.rand(1, V)
    out = searchsorted(a, v)
    assert out.shape == (B, V)

    a = torch.sort(torch.rand(B, A), dim=1)[0]
    v = torch.rand(B, V)
    out = searchsorted(a, v)
    assert out.shape == (B, V)

    a = torch.sort(torch.rand(7, A), dim=1)[0]
    v = torch.rand(9, V)
    with pytest.raises(RuntimeError):
        searchsorted(a, v)


tests = {
    'left': {
        'a': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        'v': [[-99, 99, 2], [5, 9, 8]],
        'side': 'left',
        'expected': [[0, 5, 2], [0, 4, 3]],
    },
    'right': {
        'a': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        'v': [[-99, 99, 2], [5, 9, 8]],
        'side': 'right',
        'expected': [[0, 5, 3], [1, 5, 4]],
    },
    'left-broadcast v': {
        'a': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        'v': [[-99, 99, 2]],
        'side': 'left',
        'expected': [[0, 5, 2], [0, 5, 0]],
    },
    'right-broadcast v': {
        'a': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        'v': [[-99, 99, 2]],
        'side': 'right',
        'expected': [[0, 5, 3], [0, 5, 0]],
    },
    'left-broadcast a': {
        'a': [[0, 1, 2, 3, 4]],
        'v': [[-99, 99, 2], [99, -99, 3]],
        'side': 'left',
        'expected': [[0, 5, 2], [5, 0, 3]],
    },
    'right-broadcast a': {
        'a': [[0, 1, 2, 3, 4]],
        'v': [[-99, 99, 2], [99, -99, 3]],
        'side': 'right',
        'expected': [[0, 5, 3], [5, 0, 4]],
    },
}
@pytest.mark.parametrize('test', tests.values(), ids=list(tests.keys()))
def test_correct(test, device):
    a = torch.tensor(test['a'], dtype=torch.float, device=device)
    v = torch.tensor(test['v'], dtype=torch.float, device=device)
    expected = torch.tensor(test['expected'], dtype=torch.long)

    out = searchsorted(a, v, side=test['side'])
    np.testing.assert_array_equal(out.cpu().numpy(), expected.numpy())


@pytest.mark.parametrize('Ba, Bv', [
    (Ba, Bv) for Ba, Bv in
    product([1, 150, 300], [1, 150, 300])
    if Ba == Bv or ((Ba == 1) ^ (Bv == 1))
])
@pytest.mark.parametrize('A', [1, 40, 80])
@pytest.mark.parametrize('V', [1, 40, 80])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_bigger_random(Ba, Bv, A, V, side, device):
    a = torch.sort(torch.randn(Ba, A, device=device), dim=1)[0]
    v = torch.randn(Bv, V, device=device)
    out = searchsorted(a, v, side=side)

    out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy(), side=side)
    np.testing.assert_array_equal(out.cpu().numpy(), out_np)
