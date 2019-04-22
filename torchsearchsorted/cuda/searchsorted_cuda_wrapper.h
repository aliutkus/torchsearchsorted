#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL
#include <torch/extension.h>


void searchsorted_cuda_wrapper(at::Tensor a, at::Tensor v, at::Tensor res);

#endif
