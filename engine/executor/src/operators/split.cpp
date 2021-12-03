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

#include "split.hpp"

#include "common.hpp"

namespace executor {

SplitOperator::SplitOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end()) {
    axis_ = StringToNum<int64_t>(attrs_map["axis"]);
  }
  iter = attrs_map.find("split");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&split_, attrs_map["split"], ",");
  }
}

void SplitOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  src_shape_ = input[0]->shape();
  dst_num_ = output.size();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  for (int i = 0; i < dst_num_; i++) {
    vector<int64_t> dst_shape = src_shape_;
    dst_shape[axis_] = split_[i];
    auto& dst_tensor_ptr = output[i];
    dst_tensor_ptr->set_shape(dst_shape);
    dst_tensor_ptr->set_dtype("int32");
  }
}

void SplitOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = static_cast<const int32_t*>(input[0]->data());
  // when change data value please use mutable_data
  for (int output_index = 0; output_index < dst_num_; output_index++) {
    auto dst_data = static_cast<int32_t*>(output[output_index]->mutable_data());
    vector<int64_t> dst_shape = src_shape_;
    dst_shape[axis_] = split_[output_index];
#pragma omp parallel for
    for (int j = 0; j < dst_shape[1]; j++) {
      dst_data[j] = src_data[output_index * dst_shape[1] + j];
    }
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Split);
}  // namespace executor
