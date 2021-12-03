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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "common.hpp"
#include "operator_registry.hpp"
#include "tensor.hpp"

namespace executor {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Model.
 *
 * Operator%s must implement a Forward function, in which they take their input
 * (input) Tensor%s (if any) and compute their output Tensor%s (if any).
 */
class Operator {
 public:
  explicit Operator(const OperatorConfig& conf) : operator_conf_(conf) {
    name_ = operator_conf_.name();
    type_ = operator_conf_.type();
  }

  virtual ~Operator() {}

  // use Reshape to calculate the output shape from input tensor
  virtual void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) = 0;

  inline const string& name() const { return name_; }
  inline const string& type() const { return type_; }

  inline void unref_tensors(const vector<Tensor*>& input) {
    for (size_t i = 0; i < input.size(); ++i) {
      auto status = input[i]->unref_data();
      // (TODO) maybe check the tensors
    }
  }

  virtual void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) = 0;

  virtual void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {}

  const OperatorConfig& operator_conf() const { return operator_conf_; }

 protected:
  /** The conf that stores the operator configurations */
  string name_;
  string type_;
  OperatorConfig operator_conf_;
};  // class Operator

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_
