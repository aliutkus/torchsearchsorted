#include <THC/THC.h>
#include "searchsorted_cuda_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern THCState *state;

// I don't know how to raise an exception properly in a pytorch C extension.
// doing it dirty by killing the program.
void assert(int cond, char *message)
{
  if (!cond){
    puts(message);
    exit(-1);}
}

THCudaTensor * searchsorted(THCudaTensor *a_tensor, THCudaTensor *v_tensor)
{
    // Get the number of dimensions

    assert(THCudaTensor_nDimension(state, a_tensor) == 2, "input `a` must be 2-D.\n");
    assert(THCudaTensor_nDimension(state, v_tensor) == 2, "input `v` must be 2-D.\n");

    long int nrow_a = THCudaTensor_size(state, a_tensor, 0);
    long int nrow_v = THCudaTensor_size(state, v_tensor, 0);

    assert((nrow_a==nrow_v)||(nrow_a==1)||(nrow_v==1), "`a` and `v` must have the same number of rows or one of them must have only one row.\n");

    int ncol_a = THCudaTensor_size(state, a_tensor, 1);
    int ncol_v = THCudaTensor_size(state, v_tensor, 1);


    // identify the number of rows for the result
    int nrow_res = fmax(nrow_a, nrow_v);

    // Create a result tensor of size (nrow_res, ncol_v)
    THCudaTensor *res_tensor = THCudaTensor_new(state);
    THCudaTensor_resize2d(state, res_tensor, nrow_res, ncol_v);

    // get the data of all tensors
    float *res = THCudaTensor_data(state, res_tensor);
    float *a = THCudaTensor_data(state, a_tensor);
    float *v = THCudaTensor_data(state, v_tensor);

    // get the cuda current stream
    cudaStream_t stream = THCState_getCurrentStream(state);

    // launch the cuda searchsorted function
    searchsorted_cuda(res, a, v, nrow_res, nrow_a, nrow_v, ncol_a, ncol_v, stream);
    return res_tensor;
}
