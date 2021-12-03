//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "strided_slice.hpp"

#include "common.hpp"

namespace executor {

StridedSliceOperator::StridedSliceOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  begin_mask_ = StringToNum<int64_t>(attrs_map["begin_mask"]);
  ellipsis_mask_ = StringToNum<int64_t>(attrs_map["ellipsis_mask"]);
  end_mask_ = StringToNum<int64_t>(attrs_map["end_mask"]);
  new_axis_mask_ = StringToNum<int64_t>(attrs_map["new_axis_mask"]);
  shrink_axis_mask_ = StringToNum<int64_t>(attrs_map["shrink_axis_mask"]);
  StringSplit<int64_t>(&begin_data_, attrs_map["begin"], ",");
  StringSplit<int64_t>(&end_data_, attrs_map["end"], ",");
  StringSplit<int64_t>(&strides_data_, attrs_map["strides"], ",");
}

int StridedSliceOperator::Clamp(const int v, const int lo, const int hi) {
  if (v > hi) return hi;
  if (v < lo) return lo;
  return v;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size] (or [-1, axis_size -1] if stride < 0)
// that can be used to index directly into the data.
int StridedSliceOperator::StartForAxis(const vector<int64_t>& input_shape, int axis) {
  const int axis_size = input_shape[axis];
  if (axis_size == 0) {
    return 0;
  }
  // Begin with the specified index.
  int start = begin_data_[axis];

  // begin_mask override
  if (begin_mask_ & 1 << axis) {
    if (strides_data_[axis] > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = INT_MIN;
    } else {
      // Backward iteration - use the last element.
      start = INT_MAX;
    }
  }

  // Handle negative indices
  if (start < 0) {
    start += axis_size;
  }

  // Clamping
  if (strides_data_[axis] > 0) {
    // Forward iteration
    start = Clamp(start, 0, axis_size);
  } else {
    // Backward iteration
    start = Clamp(start, -1, axis_size - 1);
  }

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
int StridedSliceOperator::StopForAxis(const vector<int64_t>& input_shape, int axis, int start_for_axis) {
  const int axis_size = input_shape[axis];
  if (axis_size == 0) {
    return 0;
  }

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask_ & (1 << axis);
  int stop = end_data_[axis];

  // When shrinking an axis, the end position does not matter (and can be
  // incorrect when negative indexing is used, see Issue #19260). Always use
  // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
  // already been adjusted for negative indices.
  if (shrink_axis) {
    return start_for_axis + 1;
  }

  // end_mask override
  if (end_mask_ & (1 << axis)) {
    if (strides_data_[axis] > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = INT_MAX;
    } else {
      // Backward iteration - use the first element.
      stop = INT_MIN;
    }
  }

  // Handle negative indices
  if (stop < 0) {
    stop += axis_size;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (strides_data_[axis] > 0) {
    // Forward iteration
    stop = Clamp(stop, 0, axis_size);
  } else {
    // Backward iteration
    stop = Clamp(stop, -1, axis_size - 1);
  }

  return stop;
}

void StridedSliceOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const auto& src_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  dst_shape_ = {};
  slice_begin_ = {};
  int64_t src_dims = src_shape.size();
  for (int i = 0; i < src_dims; ++i) {
    int64_t begin = StartForAxis(src_shape, i);
    int64_t end = StopForAxis(src_shape, i, begin);
    slice_begin_.push_back(begin);
    const bool shrink_axis = shrink_axis_mask_ & (1 << i);
    if (shrink_axis) {
      end = begin + 1;
    }
    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(strides_data_[i]));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis) {
      dst_shape_.push_back(dim_shape);
    }
  }

  // 1.3 Get tensor's adjusted strides (cached)
  dst_stride_ = GetStrides(dst_shape_);
  auto src_stride = GetStrides(src_shape);
  slice_stride_.resize(src_dims);
#pragma omp parallel for
  for (int i = 0; i < src_dims; ++i) {
    slice_stride_[i] = src_stride[i] * strides_data_[i];
  }

  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape_);
}

void StridedSliceOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = static_cast<const float*>(input[0]->data());
  // when change data value please use mutable_data
  auto dst_data = static_cast<float*>(output[0]->mutable_data());
  LOG_IF(ERROR, reinterpret_cast<void*>(dst_data) == reinterpret_cast<void*>(const_cast<float*>(src_data)))
      << "DST ptr should not be equal to SRC ptr.";

  // 1. Execute the dst
  for (int i = 0; i < dst_shape_[0]; ++i) {
#pragma omp parallel for
    for (int j = 0; j < dst_shape_[1]; ++j) {
#pragma omp simd
      for (int k = 0; k < dst_shape_[2]; ++k) {
        int dst_idx = i * dst_stride_[0] + j * dst_stride_[1] + k;
        int src_idx = slice_begin_[0] + i * slice_stride_[0] + slice_begin_[1] + j * slice_stride_[1] +
                      slice_begin_[2] + k * slice_stride_[2];
        dst_data[dst_idx] = src_data[src_idx];
      }
    }
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(StridedSlice);
}  // namespace executor
