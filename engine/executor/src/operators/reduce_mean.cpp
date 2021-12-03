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

#include "reduce_mean.hpp"

#include "common.hpp"

namespace executor {

ReduceMeanOperator::ReduceMeanOperator(const OperatorConfig& conf)
    : Operator(conf), axis_({}), dst_shape_({}), keep_dims_(true) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&axis_, attrs_map["axis"], ",");
  }
  iter = attrs_map.find("keep_dims");
  keep_dims_ = (iter != attrs_map.end() && iter->second != "false") ? true : false;
}

void ReduceMeanOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  const auto& params_shape = input[0]->shape();
  vector<int64_t> dst_shape;
  set<int64_t> axis_set;
  int64_t params_shape_size = params_shape.size();
  for (const auto& axis : axis_) {
    if (axis >= 0) {
      axis_set.insert(axis);
    } else {
      axis_set.insert(params_shape_size + axis);
    }
  }
  for (int i = 0; i < params_shape_size; ++i) {
    if (axis_set.find(i) != axis_set.end()) {
      dst_shape.push_back(1);
    } else {
      dst_shape.push_back(params_shape[i]);
      dst_shape_.push_back(params_shape[i]);  // for keep_dims is false
    }
  }

  // 1.3 Get tensor's adjusted strides
  vector<int64_t> src_stride = GetStrides(src_shape_origin);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape_origin, memory::data_type::f32, src_stride);
  memory::desc dst_md(dst_shape, memory::data_type::f32, dst_stride);

  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.2 Prepare op descriptors. last float p, float eps should be ignored in mean algorithm
  dnnl::reduction::desc reduce_mean_d(algorithm::reduction_mean, src_md, dst_md, 0.f, 0.f);

  // 2.3 Prepare primitive descriptors (cached)
  dnnl::reduction::primitive_desc reduce_mean_pd(reduce_mean_d, eng_);

  // 2.4 Prepare primitive objects (cached)
  reduce_mean_p_ = dnnl::reduction(reduce_mean_pd);

  // 2.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_);
  dst_m_ = memory(dst_md, eng_);
}

void ReduceMeanOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = input[0]->data();
  void* dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC] = src_m_;
  memory_args_[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  reduce_mean_p_.execute(s, memory_args_);

  // 5. Reshape the data of dst memory When keep_dims is false
  if (!keep_dims_) {
    output[0]->set_shape(dst_shape_);
  }

  // 6. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(ReduceMean);
}  // namespace executor
