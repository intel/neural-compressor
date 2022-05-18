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

#include "kernels/sparse_data.hpp"

namespace jd {
namespace spns {
template <typename T>
sparse_data_t<T>* reorder_to(int rows, int cols, const void* uncoded_ptr, const format_type& dst_encode_fmt) {
  const T* uncoded_data = static_cast<const T*>(uncoded_ptr);
  if (dst_encode_fmt == format_type::csr) {
    const auto& spdata = tocsr<T>(rows, cols, uncoded_data);
    return new csr_data_t<T>(spdata);
  } else if (dst_encode_fmt == format_type::csrp) {
    const auto& spdata = tocsr<T>(rows, cols, uncoded_data);
    const auto& grp = csr_with_permute<T>(rows, cols, uncoded_data);
    return new csrp_data_t<T>(spdata, format_type::csrp, grp.first, grp.second);
  }
  return nullptr;
}
template sparse_data_t<int8_t>* reorder_to<int8_t>(int, int, const void*, const format_type&);
template sparse_data_t<float>* reorder_to<float>(int, int, const void*, const format_type&);

template <typename T, dim_t group>
bsr_data_t<T>* reorder_to_bsr_amx(dim_t rows, dim_t cols, const void* uncoded_ptr) {
  const dim_t blk_row = 16;
  const dim_t blk_col = 1;
  const T* uncoded_data = static_cast<const T*>(uncoded_ptr);
  const auto bsr_data = to_bsr_amx<T, group>(rows, cols, blk_row, blk_col, uncoded_data);
  return new bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, bsr_data.indptr(), bsr_data.indices(), bsr_data.data(),
                           group);
}
template bsr_data_t<int8_t>* reorder_to_bsr_amx<int8_t, 64>(dim_t, dim_t, const void*);
template bsr_data_t<bfloat16_t>* reorder_to_bsr_amx<bfloat16_t, 32>(dim_t, dim_t, const void*);

template <typename T>
uint64_t get_uncoded_nnz(int rows, int cols, const T* uncoded_data, int line_idx) {
  if (line_idx < -1 || line_idx >= rows || uncoded_data == nullptr) {
    return 0;
  }
  int line_start = (line_idx == -1) ? 0 : line_idx;
  int line_end = (line_idx == -1) ? rows : line_idx + 1;
  uint64_t cnt = 0;
  for (int i = line_start; i < line_end; ++i) {
    for (int j = 0; j < cols; ++j) {
      cnt += is_nonzero(uncoded_data[i * cols + j]);
    }
  }
  return cnt;
}
template uint64_t get_uncoded_nnz<int8_t>(int, int, const int8_t*, int);
template uint64_t get_uncoded_nnz<float>(int, int, const float*, int);

template <typename T>
sparse_data_t<T> tocsr(int rows, int cols, const T* uncoded_data) {
  std::vector<int64_t> indptr(rows + 1);
  int all_nnz = get_uncoded_nnz(rows, cols, uncoded_data);
  std::vector<int64_t> indices(all_nnz);
  std::vector<T> data(all_nnz);
  int pos = 0;  // index of indices or data.
  for (int i = 0; i < rows; ++i) {
    indptr[i] = pos;
    for (int j = 0; j < cols; ++j) {
      auto val = uncoded_data[i * cols + j];
      if (is_nonzero(val)) {
        indices[pos] = j;
        data[pos] = val;
        ++pos;
      }
    }
    indptr[i + 1] = pos;
  }
  return std::move(sparse_data_t<T>(indptr, indices, data));
}
template sparse_data_t<int8_t> tocsr<int8_t>(int, int, const int8_t*);
template sparse_data_t<float> tocsr<float>(int, int, const float*);

template <typename T>
std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute(int rows, int cols, const T* uncoded_data) {
  auto bucket_sort = [&]() {
    std::vector<std::pair<int64_t, std::vector<int64_t>>> buckets;
    for (int i = 0; i < rows; ++i) {
      int kdim_nnz = get_uncoded_nnz(rows, cols, uncoded_data, i);
      int kdim_nnz_align = align_nnz(kdim_nnz);
      if (kdim_nnz_align == 0) {  // ignore empty rows
        continue;
      }
      auto iter = buckets.begin();
      while (iter != buckets.end()) {
        if (iter->first == kdim_nnz_align) {
          iter->second.push_back(i);
          break;
        }
        ++iter;
      }
      if (iter == buckets.end()) {
        buckets.push_back({kdim_nnz_align, {i}});
      }
    }
    return buckets;
  };
  auto buckets = bucket_sort();
  std::vector<int64_t> iperm;
  std::vector<int64_t> xgroup(1, 0);
  for (const auto& bucket : buckets) {
    const auto& permute_indices = bucket.second;
    iperm.insert(iperm.end(), permute_indices.begin(), permute_indices.end());
    xgroup.push_back(iperm.size());
  }
  return std::move(std::make_pair(iperm, xgroup));
}
template std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute<int8_t>(int, int, const int8_t*);
template std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute<float>(int, int, const float*);

template <typename T>
bsr_data_t<T> tobsr(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data) {
  std::vector<dim_t> rowptr;
  std::vector<dim_t> colidxs;
  for (dim_t b_row = 0; b_row < rows / blk_row; b_row++) {
    rowptr.push_back(colidxs.size());
    for (dim_t b_col = 0; b_col < cols / blk_col; b_col++) {
      bool is_zero = true;
      const T* dense_start = uncoded_data + b_row * blk_row * cols + b_col * blk_col;
      for (dim_t i = 0; i < blk_row; i++) {
        for (dim_t j = 0; j < blk_col; j++) {
          if (dense_start[i * cols + j] != 0) {
            is_zero = false;
            goto done_check_zero;
          }
        }
      }
    done_check_zero:
      if (!is_zero) {
        colidxs.push_back(b_col);
      }
    }
  }
  dim_t blksize = blk_row * blk_col;
  dim_t nnz = colidxs.size();
  rowptr.push_back(nnz);
  dim_t nnz_idx = 0;
  std::vector<T> data(nnz * blk_row * blk_col, 0);
  for (dim_t b_row = 0; b_row < rows / blk_row; b_row++) {
    for (dim_t b_col_idx = rowptr[b_row]; b_col_idx < rowptr[b_row + 1]; b_col_idx++, nnz_idx++) {
      dim_t b_col = colidxs[b_col_idx];
      T* blkstart = data.data() + nnz_idx * blksize;
      const T* dense_start = uncoded_data + b_row * blk_row * cols + b_col * blk_col;
      for (dim_t i = 0; i < blk_row; i++) {
        for (dim_t j = 0; j < blk_col; j++) {
          blkstart[i * blk_col + j] = dense_start[i * cols + j];
        }
      }
    }
  }
  return (bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, rowptr, colidxs, data));
}

template <typename T, dim_t group>
bsr_data_t<T> to_bsr_amx(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data) {
  bsr_data_t<T> bsr_data = tobsr<T>(rows, cols, blk_row, blk_col, uncoded_data);
  if (group == 1) {
    return bsr_data;
  }
  assert(group == 64 / sizeof(T));  // for AMX-BF16
  dim_t nrowptr = bsr_data.indptr().size();
  std::vector<dim_t> colidxs;
  std::vector<dim_t> group_rowptr(nrowptr, 0);
  for (dim_t b_row = 0; b_row < nrowptr - 1; ++b_row) {
    group_rowptr[b_row] = colidxs.size() / 32;
    dim_t b_col_idx = bsr_data.indptr()[b_row];
    while (b_col_idx < bsr_data.indptr()[b_row + 1]) {
      dim_t b_cnt = 0;
      while (b_cnt < 32 && b_col_idx < bsr_data.indptr()[b_row + 1]) {
        colidxs.push_back(bsr_data.indices()[b_col_idx++]);
        ++b_cnt;
      }
      // padding for colidxs
      while (b_cnt++ < 32) {
        colidxs.push_back(colidxs.back());
      }
    }
  }
  dim_t nnz_group = colidxs.size() / 32;
  group_rowptr[nrowptr - 1] = nnz_group;

  const dim_t blksize = blk_row * blk_col;
  std::vector<T> new_data(colidxs.size() * blksize, 0);
  dim_t data_ptr = 0;
  for (dim_t b_row = 0; b_row < nrowptr - 1; ++b_row) {
    dim_t nnz_idx = bsr_data.indptr()[b_row];
    for (dim_t group_idx = group_rowptr[b_row]; group_idx < group_rowptr[b_row + 1]; ++group_idx) {
      dim_t b_col_idx = group_idx * 32 + 1;
      dim_t b_cnt = 1;
      while (b_cnt < 32 && colidxs[b_col_idx] != colidxs[b_col_idx - 1]) {
        ++b_cnt;
        ++b_col_idx;
      }
      dim_t elem_num = b_cnt * blksize;
      for (dim_t elem = 0; elem < elem_num; elem++) {
        new_data[data_ptr + elem] = bsr_data.data()[nnz_idx * blksize + elem];
      }
      data_ptr += elem_num;
      elem_num = (32 - b_cnt) * blksize;
      data_ptr += elem_num;
      nnz_idx += 32;
    }
  }

  // reorder data to AMX layout
  std::vector<T> data(colidxs.size() * blksize, 0);
  data_ptr = 0;
  for (dim_t start_col = 0; start_col < colidxs.size(); start_col += 32) {
    for (dim_t i = 0; i < 16; ++i) {
      for (dim_t j = start_col; j < start_col + 32; ++j) {
        data[data_ptr++] = new_data[j * 16 + i];
      }
    }
  }
  return std::move(bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, group_rowptr, colidxs, data, group));
}
template bsr_data_t<bfloat16_t> to_bsr_amx<bfloat16_t, 32>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                           const bfloat16_t* uncoded_data);
template bsr_data_t<int8_t> to_bsr_amx<int8_t, 64>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                   const int8_t* uncoded_data);
}  // namespace spns
}  // namespace jd
