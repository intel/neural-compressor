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

#include "concat.hpp"

#include "common.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{{"fp32", dnnl::memory::data_type::f32},
                                                               {"s32", dnnl::memory::data_type::s32},
                                                               {"fp16", dnnl::memory::data_type::f16},
                                                               {"u8", dnnl::memory::data_type::u8},
                                                               {"s8", dnnl::memory::data_type::s8}};

ConcatOperator::ConcatOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end()) {
    axis_ = StringToNum<int64_t>(attrs_map["axis"]);
  }
}

void ConcatOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();

  // 1.2 Get tensor's number
  const int num_src = input.size();

  // 1.3 Get tensor's adjusted strides
  vector<int64_t> src_stride = GetStrides(src_shape_origin);

  // 1.4 Prepare memory descriptors
  std::vector<memory::desc> src_mds;
  std::vector<memory> src_mems;
  for (int n = 0; n < num_src; ++n) {
    auto md = memory::desc(input[n]->shape(), type2mem[input[n]->dtype()], GetStrides(input[n]->shape()));
    auto mem = memory(md, eng_);
    src_mds.push_back(md);
    src_mems.push_back(mem);
  }

  //// Part2: Derive operator's format_any memory::desc and memory.
  vector<int64_t> dst_shape;
  for (int n = 0; n < src_stride.size(); ++n) {
    if (n != axis_) {
      dst_shape.emplace_back(src_shape_origin[n]);
    } else {
      int32_t dim_sum = 0;
      for (int i = 0; i < num_src; ++i) {
        dim_sum += input[i]->shape()[n];
      }
      dst_shape.emplace_back(dim_sum);
    }
  }
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  // 2.1 Prepare primitive descriptors (cached)
  dnnl::concat::primitive_desc concat_pd(axis_, src_mds, eng_);

  // 2.2 Prepare primitive objects (cached)
  concat_p_ = dnnl::concat(concat_pd);

  // 2.3 Prepare memory objects (cached)
  src_m_ = src_mems;
  dst_m_ = memory(concat_pd.dst_desc(), eng_);
}

void ConcatOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const int num_src = input.size();

  void* dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  for (int n = 0; n < num_src; ++n) {
    const auto& src_data = input[n]->data();
    src_m_[n].set_data_handle(const_cast<void*>(src_data), s);
  }
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  std::unordered_map<int, memory> concat_args;
  for (int n = 0; n < num_src; ++n) {
    concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_m_[n]});
  }
  concat_args.insert({DNNL_ARG_DST, dst_m_});

  // 4. Execute the primitive
  concat_p_.execute(s, concat_args);

  // 5. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Concat);
}  // namespace executor
