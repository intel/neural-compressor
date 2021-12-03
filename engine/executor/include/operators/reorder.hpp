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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_REORDER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_REORDER_HPP_
#include <string>
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A Layer Normalization operator.
 *
 */

class ReorderOperator : public Operator {
 public:
  explicit ReorderOperator(const OperatorConfig& conf);
  virtual ~ReorderOperator();

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

  string output_dtype_ = "fp32";
  dnnl::primitive_attr attr_;
  bool append_sum_;
  vector<int64_t> src_perm_;
  vector<int64_t> dst_perm_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::reorder reorder_prim_;
  memory src_m_;
  memory dst_m_;

  Tensor* src_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src_min_ = nullptr;
  Tensor* src_max_ = nullptr;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_REORDER_HPP_
