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

#include "binary_add.hpp"

#include "common.hpp"

namespace executor {

BinaryAddOperator::BinaryAddOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("append_op");
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
}

void BinaryAddOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src0_shape_origin = input[0]->shape();
  const memory::dims& src1_shape_origin = input[1]->shape();

  // 1.2 Get tensor's adjusted shapes
  int64_t dsize = src0_shape_origin.size();
  // dst shape
  memory::dims dst_shape(dsize, 0);
  for (int64_t i = 0; i < dsize; ++i) {
    dst_shape[i] = max(src0_shape_origin[i], src1_shape_origin[i]);
  }

  // 1.3 Get tensor's adjusted strides
  memory::dims src0_stride = GetStrides(src0_shape_origin);
  memory::dims src1_stride = GetStrides(src1_shape_origin);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 1.4 Prepare memory descriptors
  memory::desc user_src0_md(src0_shape_origin, memory::data_type::f32, src0_stride);
  memory::desc user_src1_md(src1_shape_origin, memory::data_type::f32, src1_stride);
  memory::desc user_dst_md(dst_shape, memory::data_type::f32, dst_stride);

  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.1 Prepare format_any memory descriptors
  memory::desc any_dst_md(user_dst_md.dims(), user_dst_md.data_type(), memory::format_tag::any);

  // 2.2 Prepare op descriptors
  dnnl::binary::desc binary_d(algorithm::binary_add, user_src0_md, user_src1_md, any_dst_md);

  // 2.3 Prepare primitive descriptors (cached)
  dnnl::primitive_attr attr;
  if (append_sum_) {
    dnnl::post_ops po;
    float beta = 1.0;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }
  binary_pd_ = dnnl::binary::primitive_desc(binary_d, attr, eng_);

  // 2.4 Prepare primitive objects (cached)
  binary_p_ = dnnl::binary(binary_pd_);

  // 2.5 Prepare memory objects (cached)
  user_src0_m_ = memory(user_src0_md, eng_);
  user_src1_m_ = memory(user_src1_md, eng_);
  user_dst_m_ = memory(user_dst_md, eng_);
}

void BinaryAddOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src0_data = input[0]->data();
  const auto& src1_data = input[1]->data();
  // when change data value please use mutable_data
  // Inplace Op: 1. append_sum. 2. non-append_sum
  Tensor* dst_ptr = output[0];
  vector<Tensor*> inputs(input);
  if (!append_sum_ && (input.size() == 2) && (input[0] != nullptr) && (input[0]->size() >= dst_ptr->size())) {
    void* input_ptr = input[0]->mutable_data();
    input[0]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {input[1]};
  } else if (append_sum_ && (input.size() >= 3) && (input[2] != nullptr) && (input[2]->size() >= dst_ptr->size())) {
    void* input_ptr = input[2]->mutable_data();
    input[2]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {input[0], input[1]};
  }
  auto dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  user_src0_m_.set_data_handle(const_cast<void*>(src0_data), s);
  user_src1_m_.set_data_handle(const_cast<void*>(src1_data), s);
  user_dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  memory any_src0_m = user_src0_m_;
  memory any_src1_m = user_src1_m_;
  memory any_dst_m = user_dst_m_;

  // 2. Reorder the data when the primitive memory and user memory are different
  if (binary_pd_.dst_desc() != user_dst_m_.get_desc()) {
    any_dst_m = memory(binary_pd_.dst_desc(), eng_);
  }

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  memory_args_[DNNL_ARG_SRC_1] = any_src1_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;

  // 4. Execute the primitive
  binary_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  if (binary_pd_.dst_desc() != user_dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, user_dst_m_).execute(s, any_dst_m, user_dst_m_);
  }
  // 6. unref tensors
  this->unref_tensors(inputs);
}

REGISTER_OPERATOR_CLASS(BinaryAdd);
}  // namespace executor
