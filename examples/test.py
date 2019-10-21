import torch
from torchsearchsorted import searchsorted
import time


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
        a = torch.randn(nrows_a, nsorted_values, device='cpu')
        a = torch.sort(a, dim=1)[0]

        # generate a matrix of values to searchsort
        v = torch.randn(nrows_v, nvalues, device='cpu')

        t0 = time.time()
        test_CPU = searchsorted(a, v, test_CPU)
        print('CPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

        if not torch.cuda.is_available():
            print('CUDA is not available on this machine, cannot go further.')
            continue
        else:
            # now do the CPU
            a = a.to('cuda')
            v = v.to('cuda')

            # launch searchsorted on those
            t0 = time.time()
            test_GPU = searchsorted(a, v, test_GPU)
            print('GPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

            # compute the difference between both
            error = torch.norm(test_CPU.to('cuda').double()
                               - test_GPU.double()).cpu().numpy()

            print('    difference between CPU and GPU: %0.3f' % error)
