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

#ifndef ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace jd {
// The main kinds of kernel.
enum class kernel_kind : uint8_t {
  undef,
  sparse_matmul,
};

// The propagation kind of kernel, temporarily defined as a specific function or
// scenario. Further, the specific function can be implemented by different
// algorithms, e.g.: gemm, brgemm, ref.
enum class kernel_prop : uint8_t {
  undef,
  forward_inference,
};

// Data type.
enum class data_type : uint8_t {
  undef,
  u8,
  s8,
  fp16,
  bf16,
  fp32,
  s32,
};

// Format type.
enum class format_type : uint8_t {
  undef,
  a,
  ab,  // shape permutation = {0, 1}
  ba,  // shape permutation = {1, 0}

  // encoding format of sparse matrix
  uncoded,
  csr,
  csc,
  bsr,
  bsc,
  csrp,
};

// Engine kind.
enum class engine_kind : uint8_t {
  undef,
  cpu,
};

static std::unordered_map<data_type, int> type_size = {
    {data_type::fp32, 4}, {data_type::s32, 4}, {data_type::fp16, 2},
    {data_type::bf16, 2}, {data_type::u8, 1},  {data_type::s8, 1},
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
