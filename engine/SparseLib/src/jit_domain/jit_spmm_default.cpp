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

#include "jit_domain/jit_spmm_default.hpp"

namespace jd {
// {zmm31, zmm30, zmm29, zmm28, ...}
Xbyak::Zmm jit_spmm_default_t::TH_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm24, zmm23, zmm22, zmm21, ...}
Xbyak::Zmm jit_spmm_default_t::TW_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS - TH_;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm0, zmm1, zmm2, zmm3, ...}
Xbyak::Zmm jit_spmm_default_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = 0;
  const int& alloc_idx = alloc_start + i * TW_ + j;
  return Xbyak::Zmm(alloc_idx);
}

void jit_spmm_default_t::load_bias(const std::vector<int64_t>& m_indices) {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      vpbroadcastd(dst_tile_Vmm(i, j), ptr[reg_bias + m_indices[i] * BYTE4]);
    }
  }
}

void jit_spmm_default_t::clear_dst_tile() {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      vxorps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_default_t::load_intermediate_dst(const std::vector<int64_t>& m_indices) {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      int sliced_dst_idx = m_indices[i] * dst_stride_[0] + j * VEC;
      vmovdqu32(dst_tile_Vmm(i, j), ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4]);
    }
  }
}

void jit_spmm_default_t::handle_dst_buffer_init(int kb_idx, const std::vector<int64_t>& m_indices) {
  // Note that m_indices length is processed.
  if (kb_idx == 0) {
    if (param_.has_bias) {
      load_bias(m_indices);
    } else {
      clear_dst_tile();
    }
  } else {
    load_intermediate_dst(m_indices);
  }
}

void jit_spmm_default_t::tile_product(int tile_height, int tile_width) {
  for (int i = 0; i < tile_height; ++i) {
    for (int j = 0; j < tile_width; ++j) {
      vpdpbusd(dst_tile_Vmm(i, j), TW_Vmm(j), TH_Vmm(i));
    }
  }
}

void jit_spmm_default_t::load_dense(const std::vector<int64_t>& k_indices) {
  std::vector<int64_t> dense_rows(spns::ADJ);
  for (int i = 0; i < spns::ADJ; ++i) {
    dense_rows[i] = k_indices[i] * dst_stride_[0];
  }
  bool is_no_padded = std::find(k_indices.begin(), k_indices.end(), PADDED_NEG_ONE) == k_indices.end();
  // e.g.: when tile_shape = {4, 4}, zmm24 = b[[0, 3, 4, 9], 0:16], zmm23 = b[[0, 3, 4, 9], 16:32],
  // ..., zmm21 = b[[0, 3, 4, 9], 48:64]. zmm24 is shared by each row in a TH, so the TH's blocked.
  for (int j = 0; j < TW_; ++j) {
    int vreg_idx = TW_Vmm(j).getIdx();
    Xbyak::Xmm TW_xmm(vreg_idx);
    Xbyak::Ymm TW_ymm = Xbyak::Ymm(vreg_idx) | reg_k1;
    int vreg_temp_idx = vreg_temp.getIdx();
    Xbyak::Xmm temp_xmm(vreg_temp_idx);
    Xbyak::Ymm temp_ymm = Xbyak::Ymm(vreg_temp_idx) | reg_k1;

    if (is_no_padded) {
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[0] + j * VEC]);
      vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[1] + j * VEC]);
      vmovdqu8(temp_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[2] + j * VEC]);
      vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[3] + j * VEC]);
      vpermt2d(TW_Vmm(j), vpermt2d_arg_idx, vreg_temp);
      vpshufb(TW_Vmm(j), TW_Vmm(j), vpshufb_arg_b);
    } else {
      vxorps(vreg_temp, vreg_temp, vreg_temp);
      vxorps(TW_Vmm(j), TW_Vmm(j), TW_Vmm(j));
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[0] + j * VEC]);
      if (k_indices[1] != PADDED_NEG_ONE) {
        vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[1] + j * VEC]);
      }
      if (k_indices[2] != PADDED_NEG_ONE) {
        vmovdqu8(temp_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[2] + j * VEC]);
      }
      if (k_indices[3] != PADDED_NEG_ONE) {
        vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + dense_rows[3] + j * VEC]);
      }
      vpermt2d(TW_Vmm(j), vpermt2d_arg_idx, vreg_temp);
      vpshufb(TW_Vmm(j), TW_Vmm(j), vpshufb_arg_b);
    }
  }
}

void jit_spmm_default_t::load_sparse() {
  for (int i = 0; i < TH_; ++i) {
    vpbroadcastd(TH_Vmm(i), ptr[reg_seq_vals + (seq_pos + i) * spns::ADJ * BYTE1]);
  }
  seq_pos += TH_;
}

void jit_spmm_default_t::save_sequence_vals(const std::vector<int64_t>& m_indices,
                                            const std::unordered_map<int64_t, std::vector<int8_t>>& k_inddata_map,
                                            int pos1, int pos2) {
  for (int i = 0; i < TH_; ++i) {
    const auto& k_inddata_whole = k_inddata_map.at(m_indices[i]);
    seq_vals_.insert(seq_vals_.end(), k_inddata_whole.begin() + pos1, k_inddata_whole.begin() + pos2);
  }
}

void jit_spmm_default_t::repeat_THx4xTW_matmal(const std::vector<int64_t>& m_indices,
                                               const std::unordered_map<int64_t, std::vector<int64_t>>& k_indices_map,
                                               const std::unordered_map<int64_t, std::vector<int8_t>>& k_inddata_map) {
  int need_regs = TH_ + TW_ + TH_ * TW_ + USED_VREGS;
  LOG_IF(FATAL, need_regs >= VREG_NUMS) << "loading weight's REGs (TH=" << TH_
                                        << "), loading "
                                           "activation's REGs (TW="
                                        << TW_ << "), dst tile's REGs (TH*TW=" << (TH_ * TW_)
                                        << "). "
                                           "Their sum "
                                        << need_regs << " mustn't exceed 32zmm.";

  // ADJ=4 means 4 S8 combine a DST_S32. As ADJ repeats in K-dim, a DST_S32 also accumulates.
  // Note that a whole k-dim(segment) is processed.
  const auto& k_indices_whole = k_indices_map.begin()->second;
  int ADJ_times = ceil_div(k_indices_whole.size(), spns::ADJ);
  for (int k_repeat = 0; k_repeat < ADJ_times; ++k_repeat) {
    int ADJ_lo = k_repeat * spns::ADJ;
    int ADJ_hi = (k_repeat + 1) * spns::ADJ;
    std::vector<int64_t> k_indices(k_indices_whole.begin() + ADJ_lo, k_indices_whole.begin() + ADJ_hi);
    // Step 1: load dense (activation). Note that k_indices length is processed.
    load_dense(k_indices);
    // Step 2: load sparse (weight). Note that k_indices length is processed.
    load_sparse();
    // keep track of the nonzero values.
    save_sequence_vals(m_indices, k_inddata_map, ADJ_lo, ADJ_hi);
    // Step 3: tile product. Note that k_indices length is processed.
    // A tile product can calculate at least 1 row and 16 columns of DST.
    // Min tile calculation: Tile width/height is 1, compute (1, ADJ) x (ADJ, 16) = (1, 16) matmul.
    if (!param_.sub_func || TH_ != param_.tile_shape[0]) {
      tile_product(TH_, TW_);
    } else {
      call(sub_func_fptr_);
    }
  }
}

void jit_spmm_default_t::mul_scale(int i) {
  for (int j = 0; j < TW_; ++j) {
    vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));
    vmulps(dst_tile_Vmm(i, j), vreg_dst_temp, dst_tile_Vmm(i, j));
  }
}

void jit_spmm_default_t::move_out(int i, int j, int row_idx, int bytes) {
  int sliced_dst_idx = row_idx * dst_stride_[0] + j * VEC;
  if (bytes == BYTE1) {
    vpmovsdb(ptr[reg_dst + reg_nt_absolute_idx * bytes + sliced_dst_idx * bytes], dst_tile_Vmm(i, j));
  } else if (bytes == BYTE4) {
    vmovdqu32(ptr[reg_dst + reg_nt_absolute_idx * bytes + sliced_dst_idx * bytes], dst_tile_Vmm(i, j));
  }
}

void jit_spmm_default_t::store_intermediate_dst(const std::vector<int64_t>& m_indices) {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      int sliced_dst_idx = m_indices[i] * dst_stride_[0] + j * VEC;
      vmovdqu32(ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4], dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_default_t::handle_dst_buffer_epilogue(int kb_idx, const std::vector<int64_t>& m_indices) {
  if (kb_idx == k_blocks_ - 1) {
    for (int i = 0; i < TH_; ++i) {
      int row_idx = m_indices[i];
      vbroadcastss(vreg_dst_temp, ptr[reg_scale + row_idx * BYTE4]);  // move in scale.
      mul_scale(i);
      for (int j = 0; j < TW_; ++j) {
        if (output_type_ == data_type::u8 || output_type_ == data_type::s8) {
          vcvtps2dq(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));
          move_out(i, j, row_idx, BYTE1);
        } else if (output_type_ == data_type::fp32) {
          if (param_.append_sum) {
            int sliced_dst_idx = row_idx * dst_stride_[0] + j * VEC;
            vmovups(vreg_dst_temp, ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4]);
            vaddps(dst_tile_Vmm(i, j), vreg_dst_temp, dst_tile_Vmm(i, j));
          }
          move_out(i, j, row_idx, BYTE4);
        }
      }
    }
  } else {  // direct move out intermediate results
    store_intermediate_dst(m_indices);
  }
}

std::unordered_map<int64_t, std::vector<int64_t>> jit_spmm_default_t::get_idx_balanced(
    const std::vector<int64_t>& m_indices, const std::vector<int64_t>& sparse_indptr,
    const std::vector<int64_t>& sparse_indices, int lo, int hi) {
  std::unordered_map<int64_t, std::vector<int64_t>> k_indices_map;
  for (const auto& i : m_indices) {
    int start = sparse_indptr[i];
    int end = sparse_indptr[i + 1];
    for (int j = start; j < end; ++j) {
      int index = sparse_indices[j];
      if (index >= lo && index < hi) {
        k_indices_map[i].push_back(index);
      }
    }
  }
  // padding K-dim indices.
  int max_len = 0;
  for (const auto& p : k_indices_map) {
    int vec_size = p.second.size();
    max_len = std::max(max_len, vec_size);
  }
  int padded_len = ceil_div(max_len, spns::ADJ) * spns::ADJ;
  for (auto& p : k_indices_map) {
    auto& vec = p.second;
    for (int idx = vec.size(); idx < padded_len; ++idx) {
      vec.push_back(PADDED_NEG_ONE);
    }
  }
  return k_indices_map;
}

std::unordered_map<int64_t, std::vector<int8_t>> jit_spmm_default_t::get_val_balanced(
    const std::vector<int64_t>& m_indices, const std::vector<int64_t>& sparse_indptr,
    const std::vector<int64_t>& sparse_indices, int lo, int hi, const std::vector<int8_t>& sparse_inddata) {
  std::unordered_map<int64_t, std::vector<int8_t>> k_inddata_map;
  for (const auto& i : m_indices) {
    int start = sparse_indptr[i];
    int end = sparse_indptr[i + 1];
    for (int j = start; j < end; ++j) {
      int index = sparse_indices[j];
      int inddata = sparse_inddata[j];
      if (index >= lo && index < hi) {
        k_inddata_map[i].push_back(inddata);
      }
    }
  }
  // padding K-dim indices.
  int max_len = 0;
  for (const auto& p : k_inddata_map) {
    int vec_size = p.second.size();
    max_len = std::max(max_len, vec_size);
  }
  int padded_len = ceil_div(max_len, spns::ADJ) * spns::ADJ;
  for (auto& p : k_inddata_map) {
    auto& vec = p.second;
    for (int idx = vec.size(); idx < padded_len; ++idx) {
      vec.push_back(PADDED_ZERO);
    }
  }
  return k_inddata_map;
}

void jit_spmm_default_t::read_params() {
  mov(ebx, 0xf0);
  kmovb(reg_k1, ebx);

  mov(reg_seq_vals, ptr[param1]);
  mov(reg_dense, ptr[param1 + PTR_SIZE]);
  mov(reg_bias, ptr[param1 + 2 * PTR_SIZE]);
  mov(reg_dst, ptr[param1 + 3 * PTR_SIZE]);
  mov(reg_scale, ptr[param1 + 4 * PTR_SIZE]);
  mov(reg_nb_start, ptr[param1 + 5 * PTR_SIZE]);
  mov(reg_nb_end, ptr[param1 + 5 * PTR_SIZE + BYTE8]);
}

void jit_spmm_default_t::gen_sub_function() {
  Xbyak::util::StackFrame callee1_sf(this, 0);
  tile_product(param_.tile_shape[0], param_.tile_shape[1]);
}

void jit_spmm_default_t::params_alias(const ssd::flat_param_t& param) {
  n_blocks_ = param.mkn_blocks[2];
  nb_size_ = ceil_div(param.N, n_blocks_);

  TW_ = param.tile_shape[1];
  nt_size_ = TW_ * VEC;
  n_tiles_ = ceil_div(nb_size_, nt_size_);

  TH_ = param.tile_shape[0];
  mt_size_ = TH_ * 1;
  m_tiles_ = ceil_div(param.M, mt_size_);

  k_blocks_ = 1;
  kb_size_ = ceil_div(param.K, k_blocks_);
  dst_stride_ = {param.N, 1};
  output_type_ = param.output_type;
}

void jit_spmm_default_t::generate() {
  params_alias(param_);
  const auto& iperm = csrp_->iperm();
  const auto& avg_group = param_.avg_group;
  const auto& sparse_indptr = csrp_->indptr();
  const auto& sparse_indices = csrp_->indices();
  const auto& sparse_inddata = csrp_->data();
  inLocalLabel();  // use local label for multiple instance
  if (param_.sub_func) {
    sub_func_fptr_ = getCurr();
    gen_sub_function();
    callee_functions_code_size_ = getSize();
  }
  Xbyak::Label g_label1;
  Xbyak::Label g_label2;
  {
    Xbyak::util::StackFrame spmm_sf(this, 1, 0, stack_space_needed_);
    read_params();
    imul(reg_nb_start, reg_nb_start, nb_size_);
    imul(reg_nb_end, reg_nb_end, nb_size_);

    // initialize the control avx vectors which we are going to use for permutes and shuffles.
    vpmovzxbd(vpermt2d_arg_idx, ptr[rip + g_label1]);
    vbroadcasti32x4(vpshufb_arg_b, ptr[rip + g_label2]);

    // Loop-N1: Assembly loop at "n_blocks". Asm code fold.
    Xbyak::Label L_nb_loop;
    L(L_nb_loop);
    // When K-dim is cut into k_blocks parts, it'll produce same number of DST intermediate results.
    for (int kb_idx = 0; kb_idx < k_blocks_; ++kb_idx) {
      int kb_lo = kb_idx * kb_size_;
      int kb_hi = (kb_idx + 1) * kb_size_;
      int ngrp = avg_group.size() - 1;
      for (int i = 0; i < ngrp; ++i) {
        int i_start = avg_group[i];
        int i_end = avg_group[i + 1];
        int istart_rem = ceil_div(i_start, param_.tile_shape[0]) * param_.tile_shape[0] - i_start;
        int iend_rem = i_end % param_.tile_shape[0];
        mt_size_ = param_.tile_shape[0] * 1;
        m_tiles_ = ceil_div(i_end, mt_size_) - (i_start / mt_size_);
        int mt_pos = i_start;
        // Loop-M2: CPP loop at "m_tiles". Asm code unroll.
        for (int i2 = 0; i2 < m_tiles_; ++i2) {
          int len = mt_size_;
          TH_ = param_.tile_shape[0];
          if (i2 == 0 && istart_rem != 0) {  // If m_tiles's head block is unaligned, dynamically set TH_.
            len = istart_rem;
            TH_ = istart_rem;
          } else if (i2 == m_tiles_ - 1 &&
                     iend_rem != 0) {  // If m_tiles's tail block is unaligned, dynamically set TH_.
            len = iend_rem;
            TH_ = iend_rem;
          }
          const auto& m_indices = std::vector<int64_t>(iperm.begin() + mt_pos, iperm.begin() + mt_pos + len);
          mt_pos += len;
          xor_(reg_nt_relative_idx, reg_nt_relative_idx);
          mov(reg_nt_absolute_idx, reg_nb_start);

          // n_blocks and n_tiles are N-dim loops, and can be used for N-dim multithread.
          // n_blocks is asm outer loop, with longer assembly context span. n_tiles is inner loop.
          // Loop-N2: Assembly loop at "n_tiles". Asm code fold.
          Xbyak::Label L_nt_loop;
          L(L_nt_loop);
          // init dst buffer, like init the value to bias or the previous intermediate result.
          handle_dst_buffer_init(kb_idx, m_indices);
          auto k_indices_map = get_idx_balanced(m_indices, sparse_indptr, sparse_indices, kb_lo, kb_hi);
          auto k_inddata_map = get_val_balanced(m_indices, sparse_indptr, sparse_indices, kb_lo, kb_hi, sparse_inddata);
          repeat_THx4xTW_matmal(m_indices, k_indices_map, k_inddata_map);
          // generate the epilogue logic. This is different depending on B_blocks value (should we
          // cache intermediate results or write results with post-op to output)
          handle_dst_buffer_epilogue(kb_idx, m_indices);

          add(reg_nt_absolute_idx, nt_size_);
          add(reg_nt_relative_idx, nt_size_);
          cmp(reg_nt_relative_idx, nb_size_);
          jb(L_nt_loop, T_NEAR);  // Loop-N2 end.
        }                         // Loop-M2 end.
      }
    }
    add(reg_nb_start, nb_size_);
    cmp(reg_nb_start, reg_nb_end);
    jl(L_nb_loop, T_NEAR);  // Loop-N1 end.
  }
  int word_size = 1;
  int num_size = 16;
  const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
  const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  L(g_label1);
  for (int i = 0; i < num_size; ++i) {
    db(vpermt2d_control[i], word_size);
  }
  L(g_label2);
  for (int i = 0; i < num_size; ++i) {
    db(vpshufb_control[i], word_size);
  }
  outLocalLabel();  // end of local label
}
}  // namespace jd
