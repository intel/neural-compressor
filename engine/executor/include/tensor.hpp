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

#ifndef ENGINE_EXECUTOR_INCLUDE_TENSOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_TENSOR_HPP_
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common.hpp"
#include "conf.hpp"
#include "memory_allocator.hpp"

namespace executor {
/**
 * @brief A wrapper around Memory holders serving as the basic
 *        computational unit through which Operator%s, Model%s interact.
 *
 */
class Tensor {
 public:
  Tensor(void* data, const vector<int64_t>& shape, const string& dtype, const vector<int64_t>& strides = {},
         const vector<int64_t>& location = {}, const string& name = "")
      : name_(name), data_(data), shape_(shape), dtype_(dtype), location_(location), strides_(strides) {}

  // for pybind caster
  Tensor() : data_(nullptr), shape_({}), dtype_("fp32"), name_("") {}

  explicit Tensor(const TensorConfig& tensor_config) : data_(nullptr) {
    name_ = tensor_config.name();
    shape_ = tensor_config.shape();
    location_ = tensor_config.location();
    dtype_ = tensor_config.dtype();
    strides_ = tensor_config.strides();
  }
  // use data after set_shape
  inline const void* data() {
    if (shm_handle_ != 0) {
      data_ = MemoryAllocator::ManagedShm().get_address_from_handle(shm_handle_);
    }
    if (data_ == nullptr) {
      data_ = MemoryAllocator::get().GetMemory(this->size() * type2bytes[this->dtype()], this->life());
      // MemoryAllocator::get().SetName(data_, this->name());
    }
    return data_;
  }
  inline void* mutable_data() {
    if (shm_handle_ != 0) {
      data_ = MemoryAllocator::ManagedShm().get_address_from_handle(shm_handle_);
    }
    if (data_ == nullptr) {
      data_ = MemoryAllocator::get().GetMemory(this->size() * type2bytes[this->dtype()], this->life());
      // MemoryAllocator::get().SetName(data_, this->name());
    }
    return data_;
  }

  void set_data(void* data) {
    if (data_ != nullptr) MemoryAllocator::get().ResetMemory(data_, 0);
    auto exists_status = MemoryAllocator::get().CheckMemory(data);
    if (exists_status != -1) MemoryAllocator::get().ResetMemory(data, this->life());
    data_ = data;
  }

  int unref_data(bool inplace = false) {
    // weight tensor no need to unref
    if (!location_.empty()) return 0;
    auto status = MemoryAllocator::get().UnrefMemory(data_, inplace);
    // if we got status == -1, will keep the pointer
    if (status == 0) data_ = nullptr;
    return status;
  }

  void set_name(const string& name) { name_ = name; }
  void set_shape(const vector<int64_t>& shape) { shape_ = shape; }
  void set_dtype(const string& dtype) { dtype_ = dtype; }
  void add_tensor_life(const int count) { life_count_ += count; }

  inline size_t size() { return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>()); }

  void set_shm_handle(const ipc::managed_shared_memory::handle_t& h) { shm_handle_ = h; }
  bool is_shared() { return shm_handle_ != 0; }

  inline const string& name() const { return name_; }
  inline const int life() const { return life_count_; }
  inline const int left_life() const {
    return MemoryAllocator::get().CheckMemory(data_);
  }  // return -1 represent the data should always be hold.
  inline const void* raw_data() const { return data_; }
  inline const vector<int64_t>& shape() const { return shape_; }
  inline const vector<int64_t>& location() const { return location_; }
  inline const string& dtype() const { return dtype_; }
  inline const vector<int64_t>& strides() const { return strides_; }

 protected:
  string name_;
  void* data_;
  vector<int64_t> shape_;
  string dtype_;
  vector<int64_t> location_;
  vector<int64_t> strides_;

  // for memory handling
  int life_count_ = 0;

  // If shm_handle_ not equal to 0, which means it is on shared memory
  ipc::managed_shared_memory::handle_t shm_handle_ = 0;
};  // class Tensor
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_TENSOR_HPP_
