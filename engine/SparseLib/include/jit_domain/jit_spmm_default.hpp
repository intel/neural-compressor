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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_DEFAULT_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_DEFAULT_HPP_

#include <omp.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "jit_generator.hpp"
#include "../kernels/sparse_data.hpp"
#include "../kernels/spmm_types.hpp"
#include "utils.hpp"

namespace jd {
/**
 * @brief jit_spmm_default_t calculates this kind matmul: sparse x dense = dst.
 *        weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
 */
class jit_spmm_default_t : public jit_generator {
 public:
  explicit jit_spmm_default_t(const ssd::flat_param_t& param)
      : jit_generator(), param_(param), csrp_(param_.sparse_ptr) {}
  virtual ~jit_spmm_default_t() {}

 public:
  const void* sequence_vals() const { return seq_vals_.data(); }

 private:
  ssd::flat_param_t param_;
  csrp_data_t<int8_t>* csrp_;
  std::vector<int8_t> seq_vals_;

 private:
  void generate() override;

 private:
  // internal API of op kernel
  Xbyak::Zmm TH_Vmm(int i = 0);           // Register allocator of load weight. 1D shape=(TH)
  Xbyak::Zmm TW_Vmm(int i = 0);           // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i, int j);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)
  void params_alias(const ssd::flat_param_t& param);
  void read_params();
  void load_bias(const std::vector<int64_t>& m_indices);
  void load_dense(const std::vector<int64_t>& k_indices);
  void load_sparse();
  void tile_product(int tile_height, int tile_width);
  void handle_dst_buffer_init(int kb_idx, const std::vector<int64_t>& m_indices);
  void handle_dst_buffer_epilogue(int kb_idx, const std::vector<int64_t>& m_indices);
  void mul_scale(int i);
  void move_out(int i, int j, int row_idx, int bytes = 1);
  std::unordered_map<int64_t, std::vector<int64_t>> get_idx_balanced(const std::vector<int64_t>& m_indices,
                                                                     const std::vector<int64_t>& sparse_indptr,
                                                                     const std::vector<int64_t>& sparse_indices, int lo,
                                                                     int hi);
  std::unordered_map<int64_t, std::vector<int8_t>> get_val_balanced(const std::vector<int64_t>& m_indices,
                                                                    const std::vector<int64_t>& sparse_indptr,
                                                                    const std::vector<int64_t>& sparse_indices, int lo,
                                                                    int hi, const std::vector<int8_t>& sparse_inddata);
  void repeat_THx4xTW_matmal(const std::vector<int64_t>& m_indices,
                             const std::unordered_map<int64_t, std::vector<int64_t>>& k_indices_map,
                             const std::unordered_map<int64_t, std::vector<int8_t>>& k_inddata_map);
  void clear_dst_tile();
  void load_intermediate_dst(const std::vector<int64_t>& m_indices);
  void store_intermediate_dst(const std::vector<int64_t>& m_indices);
  void save_sequence_vals(const std::vector<int64_t>& m_indices,
                          const std::unordered_map<int64_t, std::vector<int8_t>>& k_inddata_map, int pos1, int pos2);
  void gen_sub_function();

 private:
  int64_t n_blocks_ = 0;  // The number of blocks divided in N dimension.
  int64_t nb_size_ = 0;   // The number of columns contained in a block of N dimension.
  int64_t k_blocks_ = 0;  // The number of blocks divided in K dimension.
  int64_t kb_size_ = 0;   // The number of columns contained in a block of K dimension.
  int64_t TW_ = 0;        // tile_width, its unit is different from numerical matrix.
  int64_t nt_size_ = 0;   // The number of columns contained in a tile of N dimension.
  int64_t n_tiles_ = 0;   // The number of tiles contained in a block of N dimension.
  int64_t TH_ = 0;        // tile_height, its unit is different from numerical matrix.
  int64_t mt_size_ = 0;   // The number of rows contained in a tile of M dimension.
  int64_t m_tiles_ = 0;   // The number of tiles contained in a block of M dimension.
  std::vector<int64_t> dst_stride_;
  data_type output_type_;
  const int64_t PADDED_NEG_ONE = -1;
  const int64_t PADDED_ZERO = 0;
  int64_t seq_pos = 0;
  const uint8_t* sub_func_fptr_ = nullptr;

 private:
  static constexpr int stack_space_needed_ = 200;
  static constexpr int BYTE8 = 8;
  static constexpr int BYTE4 = 4;
  static constexpr int BYTE1 = 1;
  static constexpr int VREG_NUMS = 32;
#ifdef XBYAK64
  static constexpr int PTR_SIZE = 8;
#else
  static constexpr int PTR_SIZE = 4;
#endif
  // Register decomposition
  const Xbyak::Reg64& param1 = rdi;
  const Xbyak::Reg64& reg_seq_vals = rcx;  // the first argument which is packed nonzero values pointer
  const Xbyak::Reg64& reg_dense = rdx;     // the second argument which is input matrix pointer
  const Xbyak::Reg64& reg_bias = rsi;      // the third argument which is bias values pointer
  const Xbyak::Reg64& reg_dst = rax;       // the fourth argument which is output matrix pointer
  const Xbyak::Reg64& reg_scale = rbx;     // the scale
  const Xbyak::Reg64& reg_nb_start = r8;   // start iteration count in the C dimension, useful for multithreading
  const Xbyak::Reg64& reg_nb_end = r9;     // end iteration count in the C dimension, useful for multithreading
  const Xbyak::Opmask& reg_k1 = k1;

  const Xbyak::Reg64& reg_nt_relative_idx = r10;
  const Xbyak::Reg64& reg_nt_absolute_idx = r11;

  const Xbyak::Zmm& vpermt2d_arg_idx = zmm31;
  const Xbyak::Zmm& vpshufb_arg_b = zmm30;
  const Xbyak::Zmm& vreg_temp = zmm29;
  const Xbyak::Zmm& vreg_dst_temp = vreg_temp;
  static constexpr int USED_VREGS = 3;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_DEFAULT_HPP_
