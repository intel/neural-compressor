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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_CACHE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_CACHE_HPP_
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>

#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernel_hashing.hpp"

namespace jd {
class kernel_cache {
 public:
  static kernel_cache& instance() {
    static constexpr uint64_t capacity = 1024;
    static kernel_cache inst(capacity);
    return inst;
  }
  virtual ~kernel_cache() {}

 public:
  const std::shared_ptr<const kernel_t>& find_or_construct(
      const operator_desc& op_desc, const std::function<bool(std::shared_ptr<const kernel_t>&)>& callback);
  const std::shared_ptr<const kernel_desc_t>& get_kd(const operator_desc& op_desc);

 private:
  explicit kernel_cache(int64_t capacity) : capacity_(capacity) {}
  const std::shared_ptr<const kernel_t>& get(const operator_desc& op_desc);
  void set(const std::shared_ptr<const kernel_t>& kernel);

 private:
  uint64_t capacity_;
  std::unordered_map<operator_desc, std::shared_ptr<const kernel_t>, hash_t> cache_;

  std::condition_variable cv_;
  std::mutex mtx_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_CACHE_HPP_
