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

#include "gelu.hpp"

#include "common.hpp"

namespace executor {

GeluOperator::GeluOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algorithm_ = iter->second;
  }
}

void GeluOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Prepare tensors shape and memory descriptors
  // 1.1: Prepare src tensor shape
  const memory::dims& src_shape = input[0]->shape();

  // 1.2 Set dst tensor shape
  const memory::dims& dst_shape = src_shape;
  Tensor* dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  // 1.3 Get tensor's strides
  memory::dims src_stride = GetStrides(src_shape);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape, memory::data_type::f32, src_stride);
  memory::desc dst_md(dst_shape, memory::data_type::f32, dst_stride);

  // 1.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_);
  dst_m_ = memory(dst_md, eng_);

  //// Part2: Prepare primitive
  // 2.1 Prepare op descriptors
  algorithm gelu_algorithm;
  if (algorithm_ == "gelu_erf") {
    gelu_algorithm = algorithm::eltwise_gelu_erf;
  } else if (algorithm_ == "gelu_tanh") {
    gelu_algorithm = algorithm::eltwise_gelu_tanh;
  } else {
    LOG(ERROR) << "Gelu algorithm is: " << algorithm_ << ", not supported. Only gelu_erf or gelu_tanh is supported.";
  }
  auto gelu_d = dnnl::eltwise_forward::desc(prop_kind::forward_inference, gelu_algorithm, src_md, 0.f, 0.f);

  // 2.2 Prepare primitive descriptors
  dnnl::eltwise_forward::primitive_desc gelu_pd(gelu_d, eng_);

  // 2.3 Prepare primitive objects (cached)
  gelu_p_ = dnnl::eltwise_forward(gelu_pd);
}

void GeluOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 1. Alias variables part
  const void* src_data = input[0]->data();
  void* dst_data = output[0]->mutable_data();

  // 2. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 3. Insert memory args
  unordered_map<int, memory> memory_args;
  memory_args[DNNL_ARG_SRC] = src_m_;
  memory_args[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  gelu_p_.execute(s, memory_args);

  // 5. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Gelu);
}  // namespace executor
