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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_TRANSPOSE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_TRANSPOSE_HPP_
#include <string>
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"
#include "unsupported/Eigen/CXX11/Tensor"

namespace executor {

/**
 * @brief A Transpose operator.
 *
 */

class TransposeOperator : public Operator {
 public:
  explicit TransposeOperator(const OperatorConfig& conf);
  virtual ~TransposeOperator();

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  string output_dtype_ = "fp32";
  vector<int64_t> src_perm_;
  vector<int64_t> dst_perm_;
  vector<int64_t> dst_stride_;

  Eigen::array<int, 4> perm_;
  Eigen::DSizes<Eigen::DenseIndex, 4> src_shape_;
  Eigen::DSizes<Eigen::DenseIndex, 4> dst_shape_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_TRANSPOSE_HPP_
