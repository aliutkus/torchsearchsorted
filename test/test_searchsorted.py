import pytest

import torch
import numpy as np
from torchsearchsorted import searchsorted, numpy_searchsorted
from itertools import product, repeat


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

Ba_val =[1, 100, 200]
Bv_val = [1, 100, 200]
A_val = [1, 50, 500]
V_val = [1, 12, 120]
side_val = ['left', 'right']
nrepeat = 1000
from tqdm import trange
@pytest.mark.parametrize('Ba,Bv,A,V,side', product(Ba_val, Bv_val, A_val, V_val, side_val))
def test_searchsorted_correct(Ba, Bv, A, V, side, device):
    torch.manual_seed(0)
    if (Ba != Bv):
        return
    for test in trange(nrepeat):
        a = torch.sort(torch.rand(Ba, A, device=device), dim=1)[0]
        v = torch.rand(Bv, V, device=device)
        out_np = numpy_searchsorted(a.cpu().numpy(), v.cpu().numpy(), side=side)
        out = searchsorted(a, v, side=side).cpu().numpy()
        
        if np.sum(np.abs(out-out_np)):
            # there is a difference with numpy
            def sel(data, row):
                return data[0] if data.shape[0] == 1 else data[row]
            rows, columns = np.nonzero(out != out_np)
            print(rows, columns)
            for row, column in zip(rows, columns):
                print(row, column, a.shape, v.shape)
                print(sel(out, row)[column],
                      sel(out_np, row)[column])
            #print(side, a.shape, a.flatten(), v.shape, np.nonzero(out !=out_np), '\n a: ', a[40], '\n', v[40], '\nelement:', v[out != out_np], out[out != out_np], out_np[out != out_np])
        #print('np', out_np, 'ours', out)
        np.testing.assert_array_equal(out, out_np)
