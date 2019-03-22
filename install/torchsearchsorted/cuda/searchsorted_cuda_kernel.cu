#include <iostream>
#include <climits>
#include <assert.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>




#include "searchsorted_cuda_kernel.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__
int eval(scalar_t val, scalar_t *a, int row, int col, int ncol)
{
  /* Evaluates whether a[row,col] < val <= a[row, col+1]*/

    if (col == ncol-1){
      // we are on the right border. This is the answer.
      return 0;}

    // a[row,col] <= val ?
    int is_lower = (a[row*ncol + col] < val);

    // a[row,col+1] > val ?
    int is_next_higher = (a[row*ncol + col + 1] >= val);

    if (is_lower && is_next_higher) {
      // we found the answer
        return 0;
    } else if (is_lower) {
      // answer is on the right side
        return 1;
    } else {
      // answer is on the left side
        return -1;
    }
}

template <typename scalar_t>
__device__
int binary_search(scalar_t *a, int row, scalar_t val, int ncol)
{
  /* Look for the value `val` within row `row` of matrix `a`, which
  has `ncol` columns.

  the `a` matrix is assumed sorted in increasing order, row-wise

  Returns -1 if `val` is smaller than the smallest value found within that
  row of `a`. Otherwise, return the column index `res` such that:
  a[row, col] < val <= a[row, col+1]. in case `val` is larger than the
  largest element of that row of `a`, simply return `ncol`-1. */

  //start with left at 0 and right at ncol
  int right = ncol;
  int left = 0;

  while (right >= left) {
      // take the midpoint of current left and right cursors
      int mid = left + (right-left)/2;

      // check the relative position of val: is this midpoint smaller or larger
      // than val ?
      int rel_pos = eval<scalar_t>(val, a, row, mid, ncol);

      // we found the point
      if(rel_pos == 0) {
          return mid;
      } else if (rel_pos > 0) {
        // the answer is on the right side
          left = mid;
      } else {
        // the answer is on the left side
        if (!mid)
        {
          //if we're already on the first element, we didn't find
          return -1;}
        else
        {right = mid;}
      }
  }
  return -1;
}

template <typename scalar_t>
__global__
void searchsorted_kernel(
  scalar_t *res,
  scalar_t *a,
  scalar_t *v,
  int nrow_res, int nrow_a, int nrow_v, int ncol_a, int ncol_v)
{
    // get current row and column
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // check whether we are outside the bounds of what needs be computed.
    if ((row >= nrow_res) || (col >= ncol_v)) {
      return;}

    // get the value to look for
    int row_in_v = (nrow_v==1) ? 0: row;
    int row_in_a = (nrow_a==1) ? 0: row;
    int idx_in_v = row_in_v*ncol_v+col;
    int idx_in_res = row*ncol_v+col;

    // apply binary search
    res[idx_in_res] = binary_search(a, row_in_a, v[idx_in_v], ncol_a)+1;
}


void searchsorted_cuda(
  at::Tensor a,
  at::Tensor v,
  at::Tensor res){

      // Get the dimensions
      auto nrow_a = a.size(/*dim=*/0);
      auto nrow_v = v.size(/*dim=*/0);
      auto ncol_a = a.size(/*dim=*/1);
      auto ncol_v = v.size(/*dim=*/1);

      auto nrow_res = std::max(nrow_a, nrow_v);

      // prepare the kernel configuration
      dim3 threads(ncol_v, nrow_res);
      dim3 blocks(1, 1);
      if (nrow_res*ncol_v > 1024){
         threads.x = std::min(1024, int(ncol_v));
         threads.y = floor(1024/threads.x);
         blocks.x = ceil(double(ncol_v)/double(threads.x));
         blocks.y = ceil(double(nrow_res)/double(threads.y));
      }

      AT_DISPATCH_ALL_TYPES(res.type(), "searchsorted cuda", ([&] {
        searchsorted_kernel<scalar_t><<<blocks, threads>>>(
          res.data<scalar_t>(),
          a.data<scalar_t>(),
          v.data<scalar_t>(),
          nrow_res, nrow_a, nrow_v, ncol_a, ncol_v);
      }));

  }
