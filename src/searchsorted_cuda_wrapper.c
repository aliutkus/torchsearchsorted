#include <THC/THC.h>
#include "searchsorted_cuda_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern THCState *state;

void searchsorted_cuda_wrapper(THCudaTensor *a_tensor, THCudaTensor *v_tensor, THCudaTensor *res_tensor)
{
    // Get the dimensions
    long int nrow_a = THCudaTensor_size(state, a_tensor, 0);
    long int nrow_v = THCudaTensor_size(state, v_tensor, 0);
    int ncol_a = THCudaTensor_size(state, a_tensor, 1);
    int ncol_v = THCudaTensor_size(state, v_tensor, 1);

    // identify the number of rows for the result
    int nrow_res = fmax(nrow_a, nrow_v);

    // get the data of all tensors
    float *res = THCudaTensor_data(state, res_tensor);
    float *a = THCudaTensor_data(state, a_tensor);
    float *v = THCudaTensor_data(state, v_tensor);

    // get the cuda current stream
    cudaStream_t stream = THCState_getCurrentStream(state);

    // launch the cuda searchsorted function
    searchsorted_cuda(res, a, v, nrow_res, nrow_a, nrow_v, ncol_a, ncol_v, stream);
}
