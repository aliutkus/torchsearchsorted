#include <torch/extension.h>
#include "searchsorted_cpu_wrapper.h"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

int eval(float val, float *a, int row, int col, int ncol)
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


int binary_search(float *a, int row, float val, int ncol)
{
  /* Look for the value `val` within row `row` of matrix `a`, which
  has `ncol` columns.

  the `a` matrix is assumed sorted in increasing order, row-wise

  Returns -1 if `val` is smaller than the smallest value found within that
  row of `a`. Otherwise, return the column index `res` such that:
  a[row, col] < val <= a[row, col+1]. in case `val` is larger than the
  largest element of that row of `a`, simply return `ncol`-1. */

  //start with left at 0 and right at number of columns of a
  int right = ncol;
  int left = 0;

  while (right >= left) {
      // take the midpoint of current left and right cursors
      int mid = left + (right-left)/2;

      // check the relative position of val: is this midpoint smaller or larger
      // than val ?
      int rel_pos = eval(val, a, row, mid, ncol);

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


void searchsorted_cpu_wrapper(
  at::Tensor a,
  at::Tensor v,
  at::Tensor res){

    // Get the dimensions
    auto nrow_a = a.size(/*dim=*/0);
    auto ncol_a = a.size(/*dim=*/1);
    auto nrow_v = v.size(/*dim=*/0);
    auto ncol_v = v.size(/*dim=*/1);

    auto nrow_res = fmax(nrow_a, nrow_v);

    //auto acc_v = v.accessor<float, 2>();
    //auto acc_res = res.accessor<float, 2>();

    float *a_data = a.data<float>();
    float *v_data = v.data<float>();

    for (int row=0; row<nrow_res; row++){
      for (int col=0; col<ncol_v; col++){
        // get the value to look for
        int row_in_v = (nrow_v==1) ? 0: row;
        int row_in_a = (nrow_a==1) ? 0: row;

	int idx_in_v = row_in_v*ncol_v+col;
	int idx_in_res = row*ncol_v+col;

        // apply binary search
        res.data<float>()[idx_in_res] = (
          binary_search(a_data, row_in_a, v_data[idx_in_v], ncol_a)+1);
    }}

  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("searchsorted_cpu_wrapper", &searchsorted_cpu_wrapper, "searchsorted (CPU)");
  }
