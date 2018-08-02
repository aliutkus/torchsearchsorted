import torch
from searchsorted import searchsorted
import time


nrows = 2000
nsorted_values = 3000
nkeys = 100


# generate a matrix with sorted rows
a = torch.randn(nrows, nsorted_values, device='cuda')
a.sort(dim=1)

# generate a matrix of values to searchsort
x = torch.randn(nrows, nkeys, device='cuda')

# launch searchsort on those
t0 = time.time()
test = searchsorted(a, x)
print('Searched %dx%d values in %dx%d entries in %0.3fms' %
      (nrows, nkeys, nrows, nsorted_values, 1000*(time.time()-t0)))
