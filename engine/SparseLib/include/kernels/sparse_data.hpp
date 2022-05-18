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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
#include <cassert>
#include <utility>
#include <vector>

#include "kernels/spmm_types.hpp"
#include "param_types.hpp"
#include "utils.hpp"

namespace jd {
/**
 * @brief sparse_data_t class, abstraction of a pure data class. like dense
 * tensor's data member. https://matteding.github.io/2019/04/25/sparse-matrices/
 *   https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
 *   There are two types of matrix, sparse and dense, each of which can do
 * compress or uncompress. Two concepts, that is to say, sparse/dense can do
 * compress/uncompress.
 */
template <typename T>
class sparse_data_t {
 public:
  sparse_data_t() {}
  sparse_data_t(const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices, const std::vector<T>& data)
      : indptr_(indptr), indices_(indices), data_(data) {}
  virtual ~sparse_data_t() {}

 public:
  inline const std::vector<int64_t>& indptr() const { return indptr_; }
  inline const std::vector<int64_t>& indices() const { return indices_; }
  inline const std::vector<T>& data() const { return data_; }
  virtual uint64_t getnnz(int idx = -1) const {
    if (indptr_.empty()) {
      return 0;
    }
    return (idx == -1) ? (indptr_.back() - indptr_[0]) : (indptr_[idx + 1] - indptr_[idx]);
  }

 protected:
  std::vector<int64_t> indptr_;
  std::vector<int64_t> indices_;
  std::vector<T> data_;
};

template <typename T>
class csr_data_t : public sparse_data_t<T> {
 public:
  explicit csr_data_t(const format_type& encode_fmt = format_type::csr) : sparse_data_t<T>(), encode_fmt_(encode_fmt) {}
  explicit csr_data_t(const sparse_data_t<T>& spdata, const format_type& encode_fmt = format_type::csr)
      : sparse_data_t<T>(spdata), encode_fmt_(encode_fmt) {}
  virtual ~csr_data_t() {}

 public:
  inline const format_type& encode_format() const { return encode_fmt_; }

 protected:
  format_type encode_fmt_;
};

template <typename T>
class csrp_data_t : public csr_data_t<T> {
 public:
  explicit csrp_data_t(const format_type& encode_fmt = format_type::csrp) : csr_data_t<T>(encode_fmt) {}
  csrp_data_t(const sparse_data_t<T>& spdata, const format_type& encode_fmt = format_type::csrp,
              const std::vector<int64_t>& iperm = {}, const std::vector<int64_t>& xgroup = {})
      : csr_data_t<T>(spdata, encode_fmt), iperm_(iperm), xgroup_(xgroup) {}
  virtual ~csrp_data_t() {}

 public:
  inline const std::vector<int64_t>& iperm() const { return iperm_; }
  inline const std::vector<int64_t>& xgroup() const { return xgroup_; }

 protected:
  // CSRP (CSR with permutation): that rows with the same number of nonzeros are
  // grouped together. Vectorized sparse matrix multiply for compressed row
  // storage format[C].
  // https://www.climatemodeling.org/~rmills/pubs/iccs2005.pdf Learning Sparse
  // Matrix Row Permutations for Efficient SpMM on GPU Architectures[C].
  std::vector<int64_t> iperm_;   // Here iperm is the permutation vector.
  std::vector<int64_t> xgroup_;  // xgroup points to beginning indices of groups in iperm.
};

template <typename T>
class bsr_data_t : public sparse_data_t<T> {
 public:
  explicit bsr_data_t(const std::vector<dim_t> block_size, const std::vector<dim_t> shape,
                      const std::vector<dim_t>& indptr, const std::vector<dim_t>& indices, const std::vector<T>& data,
                      const dim_t group = 1)
      : sparse_data_t<T>(indptr, indices, data), group_(group), block_size_(block_size), shape_(shape) {
    nnz_group_ = indices.size() / group_;
  }
  virtual ~bsr_data_t() {}

 public:
  inline const std::vector<dim_t> shape() const { return shape_; }
  inline const std::vector<dim_t> block_size() const { return block_size_; }
  inline const dim_t group() const { return group_; }
  inline const dim_t nnz_group() const { return nnz_group_; }

 public:
  std::vector<dim_t> shape_;
  dim_t group_;
  dim_t nnz_group_;
  std::vector<dim_t> block_size_;
};

namespace spns {
static constexpr int ADJ = 4;  // 4 is that "Multiply groups of 4 adjacent pairs..."(vpdpbusd).

inline int align_nnz(const int& a_nnz) { return ceil_div(a_nnz, ADJ) * ADJ; }

template <typename T>
sparse_data_t<T>* reorder_to(int rows, int cols, const void* uncoded_ptr, const format_type& dst_encode_fmt);

template <typename T, dim_t group>
bsr_data_t<T>* reorder_to_bsr_amx(dim_t rows, dim_t cols, const void* uncoded_ptr);

template <typename T>
uint64_t get_uncoded_nnz(int rows, int cols, const T* uncoded_data, int line_idx = -1);

template <typename T>
sparse_data_t<T> tocsr(int rows, int cols, const T* uncoded_data);

template <typename T>
std::pair<std::vector<int64_t>, std::vector<int64_t>> csr_with_permute(int rows, int cols, const T* uncoded_data);

template <typename T>
bsr_data_t<T> tobsr(dim_t rows, dim_t cols, dim_t blocksize[2], const T* uncoded_data);

template <typename T, dim_t group>
bsr_data_t<T> to_bsr_amx(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data);
}  // namespace spns
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
