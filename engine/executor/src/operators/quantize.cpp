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
  iter = attrs_map.find("channel");
  if (iter != attrs_map.end()) {
    channel_ = StringToNum<int64_t>(attrs_map["channel"]);
  }
}

QuantizeOperator::~QuantizeOperator() {}

void QuantizeOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
    is_dynamic_ = true;
  }
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
  if (is_dynamic_) {
    dst_min_->set_dtype("fp32");
    dst_max_->set_dtype("fp32");
  } else if (output_dtype_ == "u8" || output_dtype_ == "s8") {
    scales_ = GetScales(src_min_->data(), src_max_->data(), src_min_->size(), dst_->dtype());
  }
}

void QuantizeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const vector<int64_t>& src_shape = src_->shape();

  // 1.2 Set dst dtype and shape
  dst_->set_shape(src_shape);
  if (is_dynamic_ && channel_ < 0) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }
}

void QuantizeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const void* src_data = src_->data();
  void* dst_data = dst_->mutable_data();
  const float* min_data = src_min_ != nullptr ? static_cast<const float*>(src_min_->data()) : nullptr;
  if (is_dynamic_) {
    RuntimeMinmax();
    scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
    min_data = static_cast<const float*>(dst_min_->data());
  }
  // quantize
  if (src_data != nullptr && dst_data != nullptr && min_data != nullptr) {
#if __AVX512F__
    Quantize_avx512(src_->size(), dst_->dtype(), src_data, min_data, scales_, dst_data);
#else
    Quantize(src_->size(), dst_->dtype(), src_data, min_data, scales_, dst_data);
#endif
  }
  this->unref_tensors(input);
}
void QuantizeOperator::RuntimeMinmax() {
  // use onednn reduction calculate min/max
  memory::desc src_md(src_->shape(), memory::data_type::f32, GetStrides(src_->shape()));
  memory src_m(src_md, eng_);
  src_m.set_data_handle(src_->mutable_data());
  vector<int64_t> reduce_shape(dst_->shape().size(), 1);
  vector<int64_t> reduce_stride = GetStrides(reduce_shape);
  memory::desc dst_md(reduce_shape, memory::data_type::f32, reduce_stride);
  memory reduce_min(dst_md, eng_);
  memory reduce_max(dst_md, eng_);
  reduce_min.set_data_handle(dst_min_->mutable_data());
  reduce_max.set_data_handle(dst_max_->mutable_data());
  dnnl::reduction::desc reduce_min_d(algorithm::reduction_min, src_md, dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_min_pd(reduce_min_d, eng_);
  dnnl::reduction(reduce_min_pd).execute(eng_stream_, {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, reduce_min}});
  dnnl::reduction::desc reduce_max_d(algorithm::reduction_max, src_md, dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_max_pd(reduce_max_d, eng_);
  dnnl::reduction(reduce_max_pd).execute(eng_stream_, {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, reduce_max}});
}
REGISTER_OPERATOR_CLASS(Quantize);
}  // namespace executor
