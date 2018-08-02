import torch
from searchsorted import searchsorted

if __name__ == '__main__':
    import time
    import numpy as np

    nrows = 50000
    nsorted_values = 300
    nvalues = 1000

    print("Searching %dx%d values in %dx%d entries" % (nrows, nvalues,
                                                      nrows, nsorted_values))

    # generate a matrix with sorted rows
    a = torch.randn(nrows, nsorted_values, device='cuda')
    a = torch.sort(a, dim=1)[0]

    # generate a matrix of values to searchsort
    x = torch.randn(nrows, nvalues, device='cuda')

    # launch searchsort on those
    t0 = time.time()
    test_GPU = searchsorted(a, x)
    print('GPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

    t0 = time.time()
    test_CPU = np.zeros((nrows, nvalues))
    for n in range(nrows):
        test_CPU[n] = np.searchsorted(a[n], x[n])
    print('CPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

    error = torch.norm(torch.tensor(test_CPU).float().to('cuda')
                       - test_GPU).cpu().numpy()

    print('    difference:', error)
    if error > 0:
        anp = a.cpu().numpy()
        xnp = x.cpu().numpy()
        testgpu = test_GPU.cpu().numpy()
        testcpu = test_CPU

        import ipdb; ipdb.set_trace()
