#include "searchsorted_cpu_wrapper.h"
#include <stdio.h>


int64_t bisect_left(float *array, float value, int64_t left, int64_t right) {
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
  while (left < right) {
    int64_t mid = (left + right) / 2;
    if (value > array[mid]) {
      left = ++mid;
    } else {
      right = mid;
    }
  }
  return left;
}


int64_t bisect_right(float *array, float value, int64_t left, int64_t right) {
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
  while (left < right) {
    int64_t mid = (left + right) / 2;
    if (value >= array[mid]) {
      left = ++mid;
    } else {
      right = mid;
    }
  }
  return left;
}


void searchsorted_cpu_wrapper(
          at::Tensor a,
          at::Tensor v,
          at::Tensor res,
          bool side_left) {
  float *a_data = a.data_ptr<float>();
  float *v_data = v.data_ptr<float>();
  int64_t *res_data = res.data_ptr<int64_t>();

  int64_t (*bisect)(float*, float, int64_t, int64_t);
  if (side_left) {
    bisect = &bisect_left;
  } else {
    bisect = &bisect_right;
  }

  for (int64_t i = 0; i < v.size(0); i++) {
    // Search values in the range [left, right), i.e. an entire row of a
    int64_t left = i * a.stride(0);
    int64_t right = i * a.stride(0) + a.size(1);

    for (int64_t j = 0; j < v.size(1); j++) {
      // idx_v is the location of the value in the flattened tensor v
      // idx_res is the where the result will go in the flattened tensor res
      int64_t idx_v = i * v.stride(0) + j * v.stride(1);
      int64_t idx_res = i * res.stride(0) + j * res.stride(1);
      // idx is the insertion index in the flattened tensor a
      int64_t idx = (*bisect)(a_data, v_data[idx_v], left, right);
      res_data[idx_res] = idx - i * a.stride(0);
    }
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("searchsorted_cpu_wrapper", &searchsorted_cpu_wrapper, "searchsorted (CPU)");
}
