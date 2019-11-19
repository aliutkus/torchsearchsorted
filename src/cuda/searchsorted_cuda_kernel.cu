#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "searchsorted_cuda_kernel.h"


template <typename scalar_t>
__device__
int64_t bisect_left(scalar_t *array, scalar_t value, int64_t left, int64_t right) {
/**
 * Locate the insertion point of a value in a sorted array that would
 * maintain the array sorted, i.e. the index i such that:
 * array[i] <= value < array[i + 1]
 * Only the index range [right, left) is considered.
 *
 * If the value is already present in the array, the returned index would
 * insert the value to the left of any existing entry.
 * If value is < than every element, the returned index is equal to left.
 * If value is >=  than every element, the returned index is equal to right.
 */
 int64_t mid;
  while (left < right) {
    mid = (left + right) / 2;
    if (value > array[mid]) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}


template <typename scalar_t>
__device__
int64_t bisect_right(scalar_t *array, scalar_t value, int64_t left, int64_t right) {
/**
 * Locate the insertion point of a value in a sorted array that would
 * maintain the array sorted, i.e. the index i such that:
 * array[i] < value <= array[i + 1]
 * Only the index range [right, left) is considered.
 *
 * If the value is already present in the array, the returned index would
 * insert the value to the right of any existing entry.
 * If value is <= than every element, the returned index is equal to left.
 * If value is >  than every element, the returned index is equal to right.
 */
  int64_t mid;
  while (left < right) {
    mid = (left + right) / 2;
    if (value >= array[mid]) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}


template <typename scalar_t>
__global__
void searchsorted_kernel(
          at::cuda::detail::TensorInfo<scalar_t, int64_t> a,
          at::cuda::detail::TensorInfo<scalar_t, int64_t> v,
          at::cuda::detail::TensorInfo<int64_t, int64_t> res,
          bool side_left) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i >= res.sizes[0]) || (j >= res.sizes[1])) {
    return;
  }

  // Search values in the range [left, right), i.e. an entire row of a
  int64_t left = i * a.strides[0];
  int64_t right = i * a.strides[0] + a.sizes[1];

  // idx_v is the location of the value in the flattened tensor v
  // idx_res is the where the result will go in the flattened tensor res
  int64_t idx_v = i * v.strides[0] + j * v.strides[1];
  int64_t idx_res = i * res.strides[0] + j * res.strides[1];

  // idx is the insertion index in the flattened tensor a
  int64_t idx;
  /* TODO this "if" works, but would be nicer to use function pointers:
   * check side_left in searchsorted_cuda (on CPU) and pass the right pointer
   * to the kernels (on GPU), but the fact that the bisect functions are
   * templated and are defined with __device__ makes it hard to get the pointers
   * right (the address on the CPU and on the GPU are different), see
   * https://stackoverflow.com/questions/15644261/cuda-function-pointers
  */
  if (side_left) {
    idx = bisect_left(a.data, v.data[idx_v], left, right);
  } else {
    idx = bisect_right(a.data, v.data[idx_v], left, right);
  }
  res.data[idx_res] = idx - i * a.strides[0];
}

__host__
void searchsorted_cuda(
        at::Tensor a,
        at::Tensor v,
        at::Tensor res,
        bool side_left) {
  // Kernel configuration:
  // - 2D grid of size v.size(0) x v.size(1)
  // - The grid is partitioned in blocks of 256 x 4
  // - Each thread [i, j] will search for the value v[i, j] in the i-th row of a
  dim3 threads(256, 4);
  dim3 blocks(
    (v.size(0) + threads.x - 1) / threads.x,
    (v.size(1) + threads.y - 1) / threads.y
  );

  AT_DISPATCH_ALL_TYPES(a.type(), "searchsorted cuda", ([&] {
    /* Related to the comment in searchsorted_kernel, getting the address of a
     * __device__ function from a __host__ function isn't straightforward,
     * but here's a start
    */
    // int64_t (*bisect)(scalar_t*, scalar_t, int64_t, int64_t);
    // if (side_left) {
    //   bisect = &bisect_left<scalar_t>;
    // } else {
    //   bisect = &bisect_right<scalar_t>;
    // }

    searchsorted_kernel<scalar_t><<<blocks, threads>>>(
      at::cuda::detail::getTensorInfo<scalar_t, int64_t>(a),
      at::cuda::detail::getTensorInfo<scalar_t, int64_t>(v),
      at::cuda::detail::getTensorInfo<int64_t, int64_t>(res),
      side_left);
  }));
}
