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

#include "quantize.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{{"fp32", dnnl::memory::data_type::f32},
                                                               {"s32", dnnl::memory::data_type::s32},
                                                               {"fp16", dnnl::memory::data_type::f16},
                                                               {"u8", dnnl::memory::data_type::u8},
                                                               {"s8", dnnl::memory::data_type::s8}};

QuantizeOperator::QuantizeOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();

  auto iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

QuantizeOperator::~QuantizeOperator() {}

void QuantizeOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  switch (input_size) {
    case 1: {
      src_ = input[0];
      break;
    }
    case 3: {
      src_ = input[0];
      src_min_ = input[1];
      src_max_ = input[2];
      break;
    }
  }
}
void QuantizeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
  if (output_dtype_ == "u8" || output_dtype_ == "s8") {
    scales_ = GetScales(src_min_->data(), src_max_->data(), src_min_->size(), dst_->dtype());
  }
}

void QuantizeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const vector<int64_t>& src_shape = src_->shape();

  // 1.2 Set dst dtype and shape
  dst_->set_shape(src_shape);
}

void QuantizeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const void* src_data = static_cast<const float*>(src_->data());
  void* dst_data = dst_->mutable_data();
  const float* min_data = src_min_ != nullptr ? static_cast<const float*>(src_min_->data()) : nullptr;
#if __AVX512F__
  Quantize_avx512(src_->size(), dst_->dtype(), src_data, min_data, scales_, dst_data);
#else
  Quantize(src_->size(), dst_->dtype(), src_data, min_data, scales_, dst_data);
#endif
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Quantize);
}  // namespace executor
