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

#include "embeddingbag.hpp"

#include "common.hpp"

namespace executor {

EmbeddingBagOperator::EmbeddingBagOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("mode");
  embedding_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
}

void EmbeddingBagOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  indices_shape_ = input[0]->shape();
  offset_shape_ = input[1]->shape();
  weight_shape_ = input[2]->shape();
  int embedding_dim = weight_shape_[1];
  // 1.2 Get tensor's adjusted shapes
  // dst shape
  vector<int64_t> dst_shape = {offset_shape_[0], embedding_dim};
  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);
}

void EmbeddingBagOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& indices_data = static_cast<const int32_t*>(input[0]->data());
  const auto& offset_data = static_cast<const int32_t*>(input[1]->data());
  const auto& weight_data = static_cast<const float*>(input[2]->data());
  // when change data value please use mutable_data
  auto dst_data = static_cast<float*>(output[0]->mutable_data());
  for (int i = 0; i < offset_shape_[0]; ++i) {
    int end_index = (i + 1 >= offset_shape_[0]) ? indices_shape_[0] : offset_data[i + 1];
#pragma omp parallel for
    for (int k = 0; k < weight_shape_[1]; ++k) {
      float sum = 0;
      for (int j = offset_data[i]; j < end_index; ++j) {
        if (indices_data[j] < weight_shape_[0]) {
          int64_t index = indices_data[j] * weight_shape_[1] + k;
          sum += weight_data[index];
        } else {
          sum = 0;
          break;
        }
      }
      dst_data[i * weight_shape_[1] + k] = sum;
    }
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(EmbeddingBag);
}  // namespace executor
