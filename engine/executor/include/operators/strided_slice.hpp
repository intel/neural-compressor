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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_STRIDED_SLICE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_STRIDED_SLICE_HPP_
#include <vector>

#include "../operator.hpp"

namespace executor {

/**
 * @brief A StridedSlice operator.
 *
 */

class StridedSliceOperator : public Operator {
 public:
  explicit StridedSliceOperator(const OperatorConfig& conf);
  virtual ~StridedSliceOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  int Clamp(const int v, const int lo, const int hi);
  int StartForAxis(const vector<int64_t>& input_shape, int axis);
  int StopForAxis(const vector<int64_t>& input_shape, int axis, int start_for_axis);

 private:
  int64_t begin_mask_;
  int64_t ellipsis_mask_;
  int64_t end_mask_;
  int64_t new_axis_mask_;
  int64_t shrink_axis_mask_;
  vector<int64_t> begin_data_;
  vector<int64_t> end_data_;
  vector<int64_t> strides_data_;

  vector<int64_t> dst_shape_;
  vector<int64_t> dst_stride_;
  vector<int64_t> slice_begin_;
  vector<int64_t> slice_stride_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_STRIDED_SLICE_HPP_
