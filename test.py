import torch
from searchsorted import searchsorted
import time
import numpy as np

if __name__ == '__main__':
    # defining the number of tests
    ntests = 2

    # defining the problem dimensions
    nrows_a = 50000
    nrows_v = 50000
    nsorted_values = 300
    nvalues = 1000

    # defines the variables. The first run will comprise allocation, the
    # further ones will not
    test_GPU = None
    test_CPU = None

    for ntest in range(ntests):
        print("Looking for %dx%d values in %dx%d entries" % (nrows_v, nvalues,
                                                             nrows_a,
                                                             nsorted_values))

        # generate a matrix with sorted rows
        a = torch.randn(nrows_a, nsorted_values, device='cuda')
        a = torch.sort(a, dim=1)[0]

        # generate a matrix of values to searchsort
        v = torch.randn(nrows_v, nvalues, device='cuda')

        # launch searchsorted on those
        t0 = time.time()
        test_GPU = searchsorted(a, v, test_GPU)
        print('GPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

        t0 = time.time()

        # now do the CPU
        nrows_res = max(nrows_a, nrows_v)
        if test_CPU is None:
            test_CPU = np.zeros((nrows_res, nvalues))
        for n in range(nrows_res):
            test_CPU[n] = np.searchsorted(a[n if nrows_a > 1 else 0],
                                          v[n if nrows_v > 1 else 0])
        print('CPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

        # compute the difference between both
        error = torch.norm(torch.tensor(test_CPU).float().to('cuda')
                           - test_GPU).cpu().numpy()

        print('    difference:', error)
