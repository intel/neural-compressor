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

#include "layer_norm.hpp"

#include "common.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

LayerNormOperator::LayerNormOperator(const OperatorConfig& conf) : Operator(conf), weight_cached_(false) {
  auto attrs_map = operator_conf_.attributes();
  epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
}

void LayerNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();
  const memory::dims& gamma_shape_origin = input[1]->shape();
  const memory::dims& beta_shape_origin = input[2]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  memory::dims dst_shape = src_shape_origin;
  // scale & shift shape
  int64_t scale_size = gamma_shape_origin.back();
  memory::dims scale_shift_shape = {2, scale_size};

  // 1.3 Get tensor's adjusted strides
  memory::dims src_stride = GetStrides(src_shape_origin);
  memory::dims dst_stride = src_stride;

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape_origin, type2mem[input[0]->dtype()], src_stride);
  memory::desc scale_shift_md(scale_shift_shape, dnnl::memory::data_type::f32, memory::format_tag::nc);
  memory::desc dst_md(dst_shape, type2mem[output[0]->dtype()], dst_stride);

  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.1 Prepare format_any memory descriptors
  // 2.2 Prepare op descriptors
  dnnl::layer_normalization_forward::desc lnorm_d(prop_kind::forward_inference, src_md, epsilon_,
                                                  dnnl::normalization_flags::use_scale_shift);

  // 2.3 Prepare primitive descriptors
  dnnl::layer_normalization_forward::primitive_desc lnorm_pd(lnorm_d, eng_);

  // 2.4 Prepare primitive objects (cached)
  lnorm_p_ = dnnl::layer_normalization_forward(lnorm_pd);

  // 2.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_);
  dst_m_ = memory(dst_md, eng_);
  memory mean_m(lnorm_pd.mean_desc(), eng_);
  memory variance_m(lnorm_pd.variance_desc(), eng_);
  memory scale_shift_m(scale_shift_md, eng_);

  if (!weight_cached_) {
    if (input[1]->is_shared() && input[2]->is_shared()) {
      int64_t scale_shift_size = scale_shift_m.get_desc().get_size();
      string scale_shift_name = input[1]->name() + input[2]->name();
      void* scale_shift_shm_ptr =
          MemoryAllocator::ManagedShm().find_or_construct<char>(scale_shift_name.c_str())[scale_shift_size](0);
      scale_shift_m.set_data_handle(scale_shift_shm_ptr);
    }
    void* scale_shift_buf = scale_shift_m.get_data_handle();
    const auto& gamma_data = input[1]->data();
    const auto& beta_data = input[2]->data();
    std::memcpy(scale_shift_buf, gamma_data, sizeof(float) * scale_size);
    std::memcpy(reinterpret_cast<float*>(scale_shift_buf) + scale_size, beta_data, sizeof(float) * scale_size);
    weight_cached_ = true;
  }

  // 2.6 Prepare memory args (cached)
  memory_args_[DNNL_ARG_MEAN] = mean_m;
  memory_args_[DNNL_ARG_VARIANCE] = variance_m;
  memory_args_[DNNL_ARG_SCALE_SHIFT] = scale_shift_m;
}

void LayerNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = input[0]->data();
  // when change data value please use mutable_data
  // Inplace Op
  Tensor* dst_ptr = output[0];
  vector<Tensor*> inputs(input);
  if (input.size() == 3 && input[0] != nullptr && input[0]->left_life() == 1 && input[0]->size() >= dst_ptr->size()) {
    void* input_ptr = input[0]->mutable_data();
    input[0]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {input[1], input[2]};
  }
  auto dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC] = src_m_;
  memory_args_[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  lnorm_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  // 6. unref tensors
  this->unref_tensors(inputs);
}

REGISTER_OPERATOR_CLASS(LayerNorm);
}  // namespace executor
