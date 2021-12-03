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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_ONE_HOT_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_ONE_HOT_HPP_
#include <vector>

#include "../operator.hpp"

namespace executor {

/**
 * @brief A Onehot operator.
 *
 */

class OnehotOperator : public Operator {
 public:
  explicit OnehotOperator(const OperatorConfig& conf);
  virtual ~OnehotOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  int64_t axis_;
  int64_t depth_;
  int64_t on_value_;
  int64_t off_value_;
  vector<int64_t> dst_shape_;
  vector<int64_t> dst_stride_;
  vector<int64_t> reduce_shape_;
  vector<int64_t> reduce_stride_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_ONE_HOT_HPP_
