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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
#include <vector>
#include <cstdint>

namespace jd {
template <typename T> class csrp_data_t;
namespace ssd {
/**
 * @brief tensors index configuration of this kernel.
 */
static constexpr int WEI = 0;
static constexpr int SRC = 1;
static constexpr int BIAS = 2;
static constexpr int DST = 3;
static constexpr int SCALES = 4;

/**
 * @brief kernel pameters passed between kernel/primitive and jit_domain.
 */
struct flat_param_t {
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias;
  bool append_sum;
  std::vector<int64_t> mkn_blocks;
  std::vector<int64_t> tile_shape;
  int64_t start;
  int64_t end;
  // sparse weight related
  csrp_data_t<int8_t>* sparse_ptr;
  std::vector<int64_t> sub_iperm;
};

/**
 * @brief kernel data at runtime.
 */
struct flat_data_t {
  const void* ptr_seq_vals;  // sequence nonzeros of sparse weight.
  const void* ptr_dense;     // activation(K, N).
  const void* ptr_bias;      // bias(M, 1).
  void* ptr_dst;             // dst(M, N).
  const void* ptr_scales;
  int64_t start;
  int64_t end;
};
}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
