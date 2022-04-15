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
sparse_data_t<T>* reorder_to(int rows, int cols, const void* uncoded_ptr,
  const format_type& dst_encode_fmt) {
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
std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute(int rows, int cols,
  const T* uncoded_data) {
  auto bucket_sort = [&](){
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
template std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute<int8_t>(int, int,
  const int8_t*);
template std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute<float>(int, int,
  const float*);
}  // namespace spns
}  // namespace jd
