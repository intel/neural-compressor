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

#include "transpose.hpp"

namespace executor {

TransposeOperator::TransposeOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();

  auto iter = attrs_map.find("src_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src_perm_, attrs_map["src_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

TransposeOperator::~TransposeOperator() {}

void TransposeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  vector<int64_t> src_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  vector<int64_t> dst_shape_origin = src_shape;
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);

  // 1.5 Set dst shape and strides
  output[0]->set_shape(dst_shape);
  output[0]->set_dtype(output_dtype_);

  for (size_t i = 0; i < 4; ++i) {
    perm_[i] = dst_perm_[i];
    src_shape_[i] = static_cast<Eigen::DenseIndex>(input[0]->shape()[i]);
    dst_shape_[i] = static_cast<Eigen::DenseIndex>(output[0]->shape()[i]);
  }
}

// 2. inference kernel(for int8 and f32)
void TransposeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto src_data = input[0]->data();
  auto dst_data = output[0]->mutable_data();
  string dtype = output[0]->dtype();
  if (dtype == "fp32") {
    auto input = Eigen::TensorMap<Eigen::Tensor<const float, 4>>(reinterpret_cast<const float*>(src_data), src_shape_);
    auto output = Eigen::TensorMap<Eigen::Tensor<float, 4>>(reinterpret_cast<float*>(dst_data), dst_shape_);
    output = input.shuffle(perm_);
  } else if (dtype == "s8") {
    auto input = Eigen::TensorMap<Eigen::Tensor<const char, 4>>(reinterpret_cast<const char*>(src_data), src_shape_);
    auto output = Eigen::TensorMap<Eigen::Tensor<char, 4>>(reinterpret_cast<char*>(dst_data), dst_shape_);
    output = input.shuffle(perm_);
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Transpose);
}  // namespace executor
