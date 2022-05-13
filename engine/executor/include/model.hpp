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

#ifndef ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_

#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common.hpp"
#include "glog/logging.h"
#include "memory_allocator.hpp"
#include "operator.hpp"
#include "operator_registry.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

namespace executor {

/**
 * @brief Connects Operator%s together into a directed acyclic graph (DAG)
 *        specified by a ModelConfig.
 *
 */
class Model {
 public:
  explicit Model(const ModelConfig& conf, const string& weight_root);
  explicit Model(const string& conf_file, const string& weight_root);
  virtual ~Model();

  void Init(const ModelConfig& conf);
  void RemoveSharedWeight(bool is_begin = false, char* count_space_name = "RemovedCount",
                          char* count_name = "removed_count", char* space_name = "SharedWeight");
  void InitSharedWeight(char* space_name = "SharedWeight");
  ipc::managed_shared_memory::handle_t LoadSharedWeight(const string& root, const string& type,
                                                        const vector<int64_t>& shape, const vector<int64_t>& location);
  vector<Tensor>& Forward(vector<Tensor>& input_data);  // NOLINT

  void SetInput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                map<string, int>* tensor_name_to_idx);

  void SetOutput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                 map<string, int>* tensor_name_to_idx);

  inline const string& name() const { return name_; }
  inline const vector<string>& operator_names() const { return operator_names_; }
  inline const vector<string>& tensor_names() const { return tensor_names_; }
  inline const vector<shared_ptr<Operator> >& operators() const { return operators_; }
  inline const vector<Tensor*>& tensors() const { return tensors_; }

  inline int num_inputs() const { return model_input_tensors_.size(); }
  inline int num_outputs() const { return model_output_tensors_.size(); }

  inline const vector<TensorConfig*>& input_configs() const { return model_input_configs_; }

  inline vector<Tensor>& output_tensors() {
    LOG(INFO) << "Output tensor size is " << model_output_tensors_.size();
    for (int i = 0; i < model_output_tensors_.size(); ++i) {
      output_tensors_[i].set_dtype(model_output_tensors_[i]->dtype());
      auto data_buffer = model_output_tensors_[i]->data();
      auto size = model_output_tensors_[i]->size();
      // copy the data from memory to an output buffer
      if (size > output_tensors_[i].size() ||
          output_tensors_[i].size() < size * type2bytes[output_tensors_[i].dtype()]) {
        free(output_tensors_[i].mutable_data());
        void* out_buffer = malloc(size * type2bytes[output_tensors_[i].dtype()]);
        output_tensors_[i].set_data(out_buffer);
        output_tensors_[i].set_shape(model_output_tensors_[i]->shape());
        memcpy(out_buffer, data_buffer, size * type2bytes[output_tensors_[i].dtype()]);
      } else {
        void* out_buffer = output_tensors_[i].mutable_data();
        memcpy(out_buffer, data_buffer, size * type2bytes[output_tensors_[i].dtype()]);
      }
    }

    for (auto& tensor_ptr : model_output_tensors_) tensor_ptr->unref_data();
    // MemoryAllocator::get().AliveBuffer();
    return output_tensors_;
  }

 protected:
  string name_;
  string weight_root_;
  vector<shared_ptr<Operator> > operators_;
  vector<string> operator_names_;
  map<string, int> operator_name_index_;
  vector<Tensor*> tensors_;
  vector<string> tensor_names_;
  map<string, int> tensor_name_index_;

  /// input output weight vecs stores the vectors of each operator.
  vector<vector<Tensor*> > input_vecs_;
  vector<vector<Tensor*> > output_vecs_;

  vector<Tensor*> model_input_tensors_;
  vector<TensorConfig*> model_input_configs_;
  vector<Tensor*> model_output_tensors_;
  vector<Tensor> output_tensors_;
  bool multi_stream_flag = (getenv("MULTI_STREAM") != NULL);
  // collect the op index with parallel thread
  unordered_map<int, int64_t> multi_stream_tasks_;
  ThreadPool tp;
  std::mutex rmutex_;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_
