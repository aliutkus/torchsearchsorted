#include <THC/THC.h>
#include "searchsorted_cuda_kernel.h"
#include <stdio.h>
#include <stdlib.h>

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

    assert(THCudaTensor_nDimension(state, a_tensor) == 2, "input `a` must be matrix.\n");
    assert(THCudaTensor_nDimension(state, v_tensor) == 2, "input `v` must be matrix.\n");

    long int nrow = THCudaTensor_size(state, a_tensor, 0);
    long int nrow_v = THCudaTensor_size(state, v_tensor, 0);

    assert(nrow==nrow_v, "`a` and `v` must have the same number of rows.\n");

    int ncol_a = THCudaTensor_size(state, a_tensor, 1);
    int ncol_v = THCudaTensor_size(state, v_tensor, 1);


    // Create a result tensor of the same size as `v_tensor`
    THCudaTensor *res_tensor = THCudaTensor_newWithTensor(state, v_tensor);

    // get the data of all tensors
    float *res = THCudaTensor_data(state, res_tensor);
    float *a = THCudaTensor_data(state, a_tensor);
    float *v = THCudaTensor_data(state, v_tensor);

    // get the cuda current stream
    cudaStream_t stream = THCState_getCurrentStream(state);

    // launch the cuda searchsorted function
    searchsorted_cuda(res, a, v, nrow, ncol_a, ncol_v, stream);
    return res_tensor;
}
