#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif


void searchsorted_cuda(float *res, float *a, float *v, int nrow_res, int nrow_a, int nrow_v, int ncol_a, int ncol_v, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
