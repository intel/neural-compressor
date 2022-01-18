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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_MERGED_EMBEDDINGBAG_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_MERGED_EMBEDDINGBAG_HPP_
#include <assert.h>

#include <string>
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"

namespace executor {

/**
 * @brief A Padding Sequence Mask operator.
 *
 */

class MergedEmbeddingbagOperator : public Operator {
 public:
  explicit MergedEmbeddingbagOperator(const OperatorConfig& conf);
  virtual ~MergedEmbeddingbagOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  string mode_;
};

template <typename T>
void emb_pooling_ker(T* out, T* in, const size_t pool_begin, const size_t pool_end, const size_t vector_size,
                     int32_t* indices_data, const string& mode);

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_MERGED_EMBEDDINGBAG_HPP_
