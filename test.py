import torch
from searchsorted import searchsorted

if __name__ == '__main__':
    import time
    import numpy as np

    nrows_a = 50000
    nrows_v = 50000
    nsorted_values = 300
    nvalues = 1000

    for ntest in range(1):
        print("Looking for %dx%d values in %dx%d entries" % (nrows_v, nvalues,
                                                             nrows_a,
                                                             nsorted_values))

        # generate a matrix with sorted rows
        a = torch.randn(nrows_a, nsorted_values, device='cuda')
        a = torch.sort(a, dim=1)[0]

        # generate a matrix of values to searchsort
        v = torch.randn(nrows_v, nvalues, device='cuda')

        # launch searchsort on those
        t0 = time.time()
        test_GPU = searchsorted(a, v)
        print('GPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

        t0 = time.time()
        nrows_res = max(nrows_a, nrows_v)
        test_CPU = np.zeros((nrows_res, nvalues))
        for n in range(nrows_res):
            test_CPU[n] = np.searchsorted(a[n if nrows_a > 1 else 0],
                                          v[n if nrows_v > 1 else 0])
        print('CPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

        error = torch.norm(torch.tensor(test_CPU).float().to('cuda')
                           - test_GPU).cpu().numpy()

        print('    difference:', error)
