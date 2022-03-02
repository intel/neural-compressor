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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONVOLUTION_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONVOLUTION_HPP_
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A Convolution operator.
 *
 */

class ConvolutionOperator : public Operator {
 public:
  explicit ConvolutionOperator(const OperatorConfig& conf);
  virtual ~ConvolutionOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

  bool weight_cached_;
  bool has_bias_;
  bool format_any_;
  bool append_sum_;
  bool binary_add_;
  bool gelu_erf_;
  bool gelu_tanh_;
  bool gelu_split_;
  bool tanh_;
  bool sigmoid_;
  bool relu_;

  bool append_eltwise_;
  float output_scale_ = 1.f;
  string output_dtype_ = "fp32";
  vector<int64_t> src_perm_;
  vector<int64_t> dst_perm_;
  int64_t group_ = 1;
  vector<int64_t> pads_;
  vector<int64_t> strides_;
  vector<int64_t> weight_shape_;

  dnnl::primitive_attr attr_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  dnnl::convolution_forward::primitive_desc convolution_pd_;
  dnnl::convolution_forward convolution_p_;
  unordered_map<int, memory> memory_args_;

  dnnl::engine gelu_eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream gelu_eng_stream_ = dnnl::stream(gelu_eng_);
  dnnl::eltwise_forward::primitive_desc gelu_pd_;
  dnnl::eltwise_forward gelu_p_;
  unordered_map<int, memory> gelu_memory_args_;

  memory::desc weight_md_;
  memory::desc any_weight_md_;
  memory::desc bias_md_;
  memory::desc any_bias_md_;
  memory src_m_;
  memory weight_m_;
  memory bias_m_;
  memory dst_m_;
  memory gelu_m_;
  memory binary_m_;

  Tensor* src_ = nullptr;
  Tensor* weight_ = nullptr;
  Tensor* bias_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src_min_ = nullptr;
  Tensor* src_max_ = nullptr;

  Tensor* weight_min_ = nullptr;
  Tensor* weight_max_ = nullptr;

  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONVOLUTION_HPP_
