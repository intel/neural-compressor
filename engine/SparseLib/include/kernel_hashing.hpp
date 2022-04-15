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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include "param_types.hpp"
#include "tensor_config.hpp"
#include "operator_config.hpp"
#include "engine.hpp"

namespace jd {
/**
 * @brief The hash function of a specific kernel descriptor or kernel primitive.
 */
class hash_t {
 public:
  uint64_t operator()(const operator_config& key) const {
    uint64_t seed = 0;
    // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
    hash_combine(seed, static_cast<uint64_t>(key.kernel_kind()));
    hash_combine(seed, static_cast<uint64_t>(key.kernel_hypotype()));
    hash_combine(seed, static_cast<uint64_t>(key.engine_kind()));
    hash_combine(seed, static_cast<uint64_t>(key.impl_nthr()));
    hash_combine(seed, get_tensor_cfgs_hash(key.tensor_cfgs()));
    hash_combine(seed, get_attr_hash(key.attrs(), key.kernel_kind()));
    return seed;
  }

 private:
  // http://boost.sourceforge.net/doc/html/boost/hash_combine.html
  template<typename T>
  static void hash_combine(size_t& seed, const T& v) {   // NOLINT
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

 private:
  uint64_t get_tensor_cfgs_hash(const std::vector<tensor_config>& t_cfgs) const {
    uint64_t seed = 0;
    int tensor_cnt = t_cfgs.size();
    for (int idx = 0; idx < tensor_cnt; ++idx) {
      for (const auto& dim : t_cfgs[idx].shape()) {
        hash_combine(seed, static_cast<uint64_t>(dim));
      }
      hash_combine(seed, static_cast<uint64_t>(t_cfgs[idx].dtype()));
      hash_combine(seed, static_cast<uint64_t>(t_cfgs[idx].ftype()));
    }
    return seed;
  }

  uint64_t get_attr_hash(const std::unordered_map<std::string, std::string>& attrs, const kernel_kind& ker_kind) const {
    auto op_attrs = attrs;
    uint64_t seed = 0;
    hash_combine(seed, op_attrs["append_sum"]);
    switch (ker_kind) {
      case kernel_kind::undef:
        break;
      case kernel_kind::sparse_matmul:
        hash_combine(seed, op_attrs["sparse_ptr"]);
        hash_combine(seed, op_attrs["mkn_blocks"]);
        hash_combine(seed, op_attrs["tile_shape"]);
        break;
      default:
        break;
    }
    return seed;
  }
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
