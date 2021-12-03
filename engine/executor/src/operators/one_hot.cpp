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

#include "one_hot.hpp"

#include "common.hpp"

namespace executor {

OnehotOperator::OnehotOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : -1;
  iter = attrs_map.find("depth");
  depth_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 2;
  iter = attrs_map.find("on_value");
  on_value_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 1;
  iter = attrs_map.find("off_value");
  off_value_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
}

void OnehotOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const auto& indices_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  switch (indices_shape.size()) {
    case 0:  // scalar
      dst_shape_ = {depth_};
      reduce_shape_ = {1};
      break;
    case 1:  // vector
      if (axis_ == -1) {
        dst_shape_ = {indices_shape[0], depth_};
        reduce_shape_ = {indices_shape[0], 1};
      } else if (axis_ == 0) {
        dst_shape_ = {depth_, indices_shape[0]};
        reduce_shape_ = {1, indices_shape[0]};
      }
      break;
    case 2:  // matrix
      if (axis_ == -1) {
        dst_shape_ = {indices_shape[0], indices_shape[1], depth_};
        reduce_shape_ = {indices_shape[0], indices_shape[1], 1};
      } else if (axis_ == 1) {
        dst_shape_ = {indices_shape[0], depth_, indices_shape[1]};
        reduce_shape_ = {indices_shape[0], 1, indices_shape[1]};
      } else if (axis_ == 0) {
        dst_shape_ = {depth_, indices_shape[0], indices_shape[1]};
        reduce_shape_ = {1, indices_shape[0], indices_shape[1]};
      }
      break;
  }

  // 1.3 Get tensor's adjusted strides (cached)
  dst_stride_ = GetStrides(dst_shape_);
  reduce_stride_ = GetStrides(reduce_shape_);

  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape_);
}

void OnehotOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& indices_data = static_cast<const int32_t*>(input[0]->data());
  // when change data value please use mutable_data
  auto dst_data = static_cast<float*>(output[0]->mutable_data());
  LOG_IF(ERROR, reinterpret_cast<void*>(dst_data) == reinterpret_cast<void*>(const_cast<int32_t*>(indices_data)))
      << "DST ptr should not be equal to SRC ptr.";

  // 1. Execute the dst
  auto do_onehot_2d = [&](const vector<int64_t>& dst_shape, const vector<int64_t>& dst_stride,
                          const vector<int64_t>& reduce_shape, const vector<int64_t>& reduce_stride, int64_t dep,
                          float*& dst_data) {
    vector<int64_t> i(2);
    // #pragma omp parallel for
    for (i[0] = 0; i[0] < dst_shape[0]; ++i[0]) {
      // #pragma omp simd
      for (i[1] = 0; i[1] < dst_shape[1]; ++i[1]) {
        auto dst_idx = i[0] * dst_stride[0] + i[1];
        auto src_idx = min(i[0], reduce_shape[0] - 1) * reduce_stride[0] + min(i[1], reduce_shape[1] - 1);
        dst_data[dst_idx] = (i[dep] == indices_data[src_idx]) ? on_value_ : off_value_;
      }
    }
  };
  auto do_onehot_3d = [&](const vector<int64_t>& dst_shape, const vector<int64_t>& dst_stride,
                          const vector<int64_t>& reduce_shape, const vector<int64_t>& reduce_stride, int64_t dep,
                          float*& dst_data) {
    vector<int64_t> i(3);
    // #pragma omp parallel for
    for (i[0] = 0; i[0] < dst_shape[0]; ++i[0]) {
      // #pragma omp simd
      for (i[1] = 0; i[1] < dst_shape[1]; ++i[1]) {
        // #pragma omp simd
        for (i[2] = 0; i[2] < dst_shape[2]; ++i[2]) {
          auto dst_idx = i[0] * dst_stride[0] + i[1] * dst_stride[1] + i[2];
          auto src_idx = min(i[0], reduce_shape[0] - 1) * reduce_stride[0] +
                         min(i[1], reduce_shape[1] - 1) * reduce_stride[1] + min(i[2], reduce_shape[2] - 1);
          dst_data[dst_idx] = (i[dep] == indices_data[src_idx]) ? on_value_ : off_value_;
        }
      }
    }
  };

  int64_t dep;
  switch (input[0]->shape().size()) {
    case 0:  // scalar
      for (int d = 0; d < depth_; ++d) {
        auto dst_idx = d;
        auto key = indices_data[0];
        dst_data[dst_idx] = (d == key) ? on_value_ : off_value_;
      }
      break;
    case 1:  // vector
      dep = (axis_ == -1) ? 1 : axis_;
      do_onehot_2d(dst_shape_, dst_stride_, reduce_shape_, reduce_stride_, dep, dst_data);
      break;
    case 2:  // matrix
      dep = (axis_ == -1) ? 2 : axis_;
      do_onehot_3d(dst_shape_, dst_stride_, reduce_shape_, reduce_stride_, dep, dst_data);
      break;
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Onehot);
}  // namespace executor
