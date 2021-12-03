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

#include "token_type_ids.hpp"

#include "common.hpp"

namespace executor {

TokenTypeIdsOperator::TokenTypeIdsOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("mode");
  if (iter != attrs_map.end()) {
    mode_ = iter->second;
  }
}

void TokenTypeIdsOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> src_shape = input[0]->shape();
  output[0]->set_shape(src_shape);
}

void TokenTypeIdsOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* slice = input[1];
  Tensor* dst = output[0];

  const vector<int64_t> dst_shape = dst->shape();
  const int batch_size = dst_shape[0];
  const int seq_len = dst_shape[1];

  int64_t slice_size = slice->size();
  assert(slice_size >= seq_len);
  const int32_t* slice_data = static_cast<const int32_t*>(slice->data());
  int32_t* dst_data = static_cast<int32_t*>(dst->mutable_data());

  if (mode_ == "roberta") {
#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
#pragma omp simd
      for (int j = 0; j < seq_len; j++) {
        dst_data[i * seq_len + j] = slice_data[j];
      }
    }
  } else {
    LOG(ERROR) << "TokenTypeIds mode is: " << mode_ << ", not supported. Only roberta is supported.";
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(TokenTypeIds);
}  // namespace executor
