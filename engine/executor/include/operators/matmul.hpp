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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
#include <string>
#include <unordered_map>
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {

using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

// \brief Matmul or Batchmatmul operators
class MatmulOperator : public Operator {
 public:
  explicit MatmulOperator(const OperatorConfig& conf);
  virtual ~MatmulOperator();

 public:
  // void ParseOperatorConfig();
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  // Matrix can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it).
  bool is_asymm_;
  bool format_any_;
  bool append_sum_;
  bool gelu_erf_;
  bool gelu_tanh_;
  bool tanh_;
  bool append_eltwise_;
  bool cache_weight_;
  bool binary_add_;
  float output_scale_ = 1.f;
  string output_dtype_ = "fp32";
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  vector<int64_t> dst_perm_;

  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  dnnl::matmul::primitive_desc matmul_pd_;
  dnnl::matmul matmul_p_;
  unordered_map<int, memory> memory_args_;

  memory src0_m_;
  memory src1_m_;
  memory bias_m_;
  memory dst_m_;
  memory binary_m_;

  Tensor* src0_ = nullptr;
  Tensor* src1_ = nullptr;
  Tensor* bias_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src0_min_ = nullptr;
  Tensor* src0_max_ = nullptr;

  Tensor* src1_min_ = nullptr;
  Tensor* src1_max_ = nullptr;

  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
