//  Copyright(c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0(the "License");
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

#include "sparse_inner_product.hpp"

namespace executor {

template <typename T>
float GetSparseRatio(const T* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize) {
  const int64_t blocknum = (shape[0] / blocksize[0]) * (shape[1] / blocksize[1]);
  int64_t zero_count = blocknum;
  for (int64_t b_row = 0; b_row < shape[0] / blocksize[0]; b_row++) {
    for (int64_t b_col = 0; b_col < shape[1] / blocksize[1]; b_col++) {
      const T* dense_start = data + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
      bool not_zero = false;
      for (int64_t i = 0; i < blocksize[0]; i++) {
        for (int64_t j = 0; j < blocksize[1]; j++) {
          if (dense_start[i * shape[1] + j] != 0) {
            zero_count--;
            not_zero = true;
            break;
          }
        }
        if (not_zero) {
          break;
        }
      }
    }
  }
  float zero_ratio = blocknum == 0 ? 0 : static_cast<float>(zero_count) / blocknum;
  return zero_ratio;
}
template float GetSparseRatio(const float* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize);
template float GetSparseRatio(const int8_t* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize);

template <typename T>
void TransposeMatrix(const T* input, const vector<int64_t>& shape, T* output) {
#pragma omp parallel for
  for (int i = 0; i < shape[0]; i++) {
#pragma omp simd
    for (int j = 0; j < shape[1]; j++) {
      output[j * shape[0] + i] = input[i * shape[1] + j];
    }
  }
}
template void TransposeMatrix(const float* input, const vector<int64_t>& shape, float* output);
template void TransposeMatrix(const int8_t* input, const vector<int64_t>& shape, int8_t* output);

// The shape of dense_matrix is given by `shape[2]`.
// Sparse block granularity is given by `blocksize[2]`.
template <typename T>
BSRMatrix<T>* create_bsr_matrix(const T* dense_matrix, const vector<int64_t>& shape, const vector<int64_t>& blocksize) {
  const int64_t blksize = blocksize[0] * blocksize[1];
  BSRMatrix<T>* bsr_matrix = new BSRMatrix<T>;
  bsr_matrix->shape = shape;
  bsr_matrix->blocksize = blocksize;
  assert(shape[0] % blocksize[0] == 0);
  assert(shape[1] % blocksize[1] == 0);
  // Initialize rowptr and colidxs arrays
  // Dynamic arrays that will be copied to bsr_matrix after initialization
  std::vector<int64_t> rowptr;
  std::vector<int64_t> colidxs;
  for (int64_t b_row = 0; b_row < bsr_matrix->shape[0] / blocksize[0]; b_row++) {
    rowptr.push_back(colidxs.size());
    for (int64_t b_col = 0; b_col < bsr_matrix->shape[1] / blocksize[1]; b_col++) {
      bool not_zero = true;
      const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
      for (int64_t i = 0; i < bsr_matrix->blocksize[0]; i++) {
        for (int64_t j = 0; j < bsr_matrix->blocksize[1]; j++) {
          if (dense_start[i * shape[1] + j] != 0) {
            not_zero = false;
            goto done_check_zero;
          }
        }
      }
    done_check_zero:
      if (!not_zero) {
        colidxs.push_back(b_col);
      }
    }
  }
  rowptr.push_back(colidxs.size());
  // init bsr_matrix->rowptr array
  bsr_matrix->nrowptr = rowptr.size();
  bsr_matrix->rowptr = new int64_t[rowptr.size()];
  for (int64_t i = 0; i < bsr_matrix->nrowptr; i++) {
    bsr_matrix->rowptr[i] = rowptr[i];
  }
  // init bsr_matrix->colidxs array
  bsr_matrix->nnz = colidxs.size();
  bsr_matrix->colidxs = new int64_t[colidxs.size()];
  for (int64_t i = 0; i < bsr_matrix->nnz; i++) {
    bsr_matrix->colidxs[i] = colidxs[i];
  }
  int64_t nnz_idx = 0;
  bsr_matrix->data = reinterpret_cast<T*>(aligned_alloc(64, (bsr_matrix->nnz * blksize * sizeof(T) / 64 + 1) * 64));
  for (int64_t b_row = 0; b_row < bsr_matrix->nrowptr - 1; b_row++) {
    for (int64_t b_col_idx = bsr_matrix->rowptr[b_row]; b_col_idx < bsr_matrix->rowptr[b_row + 1];
         b_col_idx++, nnz_idx++) {
      int64_t b_col = bsr_matrix->colidxs[b_col_idx];
      T* blkstart = bsr_matrix->data + nnz_idx * blksize;
      const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
      for (int64_t i = 0; i < bsr_matrix->blocksize[0]; i++) {
        for (int64_t j = 0; j < bsr_matrix->blocksize[1]; j++) {
          blkstart[i * bsr_matrix->blocksize[1] + j] = dense_start[i * shape[1] + j];
        }
      }
    }
  }
  return bsr_matrix;
}
template BSRMatrix<float>* create_bsr_matrix(const float* dense_matrix, const vector<int64_t>& shape,
                                             const vector<int64_t>& blocksize);
template BSRMatrix<int8_t>* create_bsr_matrix(const int8_t* dense_matrix, const vector<int64_t>& shape,
                                              const vector<int64_t>& blocksize);

template <typename T>
BSCMatrix<T>* create_bsc_matrix(const T* dense_matrix, const vector<int64_t>& shape, const vector<int64_t>& blocksize) {
  BSRMatrix<T>* bsr = create_bsr_matrix(dense_matrix, shape, blocksize);
  BSCMatrix<T>* bsc = new BSCMatrix<T>;
  const int64_t bs = blocksize[0] * blocksize[1];
  bsc->shape = bsr->shape;
  bsc->blocksize = bsr->blocksize;
  bsc->nnz = bsr->nnz;
  bsc->ncolptr = bsr->shape[1] / bsr->blocksize[1] + 1;
  bsc->data = reinterpret_cast<T*>(aligned_alloc(64, (bsr->nnz * bs * sizeof(T) / 64 + 1) * 64));
  bsc->colptr = new int64_t[bsc->ncolptr];
  bsc->rowidxs = new int64_t[bsr->nnz];
  int64_t b_col = 0, ptr = 0;
  for (; b_col < bsc->ncolptr - 1; b_col++) {
    bsc->colptr[b_col] = ptr;
    for (int64_t b_row = 0, nnz_idx = 0; b_row < bsr->nrowptr - 1; b_row++) {
      for (int64_t b_col_idx = bsr->rowptr[b_row]; b_col_idx < bsr->rowptr[b_row + 1]; b_col_idx++, nnz_idx++) {
        if (b_col == bsr->colidxs[b_col_idx]) {
          memcpy(bsc->data + ptr * bs, bsr->data + nnz_idx * bs, sizeof(T) * bs);
          bsc->rowidxs[ptr++] = b_row;
        }
      }
    }
  }
  bsc->colptr[b_col] = ptr;
  destroy_bsr_matrix(bsr);
  return bsc;
}
template BSCMatrix<float>* create_bsc_matrix(const float* dense_matrix, const vector<int64_t>& shape,
                                             const vector<int64_t>& blocksize);
template BSCMatrix<int8_t>* create_bsc_matrix(const int8_t* dense_matrix, const vector<int64_t>& shape,
                                              const vector<int64_t>& blocksize);

template <typename T>
void destroy_bsr_matrix(BSRMatrix<T>* bsr_matrix) {
  free(bsr_matrix->data);
  delete[] bsr_matrix->colidxs;
  delete[] bsr_matrix->rowptr;
  delete bsr_matrix;
}
template void destroy_bsr_matrix(BSRMatrix<float>* bsr_matrix);
template void destroy_bsr_matrix(BSRMatrix<int8_t>* bsr_matrix);

template <typename T>
void destroy_bsc_matrix(BSCMatrix<T>* bsc_matrix) {
  free(bsc_matrix->data);
  delete[] bsc_matrix->rowidxs;
  delete[] bsc_matrix->colptr;
  delete bsc_matrix;
}
template void destroy_bsc_matrix(BSCMatrix<float>* bsc_matrix);
template void destroy_bsc_matrix(BSCMatrix<int8_t>* bsc_matrix);

// int8 kernel test
void reorder_bsc_int8_4x16(BSCMatrix<int8_t>* bsc) {
  const int block_row = bsc->blocksize[0];
  const int block_col = bsc->blocksize[1];
  const int block_size = block_row * block_col;
  int8_t* new_data = reinterpret_cast<int8_t*>(aligned_alloc(64, bsc->nnz * block_size * sizeof(int8_t)));
  // reorder data
  const int8_t* block_start = bsc->data;
  int new_data_ptr = 0;
  for (int nnz_idx = 0; nnz_idx < bsc->nnz; ++nnz_idx) {
    for (int col = 0; col < bsc->blocksize[1]; ++col) {
      for (int row = 0; row < bsc->blocksize[0]; ++row) {
        new_data[new_data_ptr++] = block_start[row * block_col + col];
      }
    }
    block_start += block_size;
  }
  // currently rowidxs are from block perspective, modify them to represent the begin row in each block
  for (int i = 0; i < bsc->nnz; ++i) {
    bsc->rowidxs[i] *= block_row;
  }
  bsc->data = new_data;
}

#if __AVX512F__
/********* fp32 kernel *********/
void sparse_gemm_bsc_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B, const int64_t* rowidxs,
                         const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize, float* C,
                         const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
  __m512 d = _mm512_setzero_ps();
#pragma omp parallel for collapse(2)
  for (int64_t mb = 0; mb < M / M_NBLK; mb++) {
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      for (int64_t i = 0; i < M_NBLK; i++) {
        output[i] = _mm512_setzero_ps();
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int64_t i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < M_NBLK; i++) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_setzero_ps();
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
}

void sparse_gemm_bsc_bias_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B, const int64_t* rowidxs,
                              const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                              const float* bias, float* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
#pragma omp parallel for collapse(2)
  for (int64_t mb = 0; mb < M / M_NBLK; mb++) {
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      for (int64_t i = 0; i < M_NBLK; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int64_t i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < M_NBLK; i++) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
}

void sparse_gemm_bsc_bias_relu_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                   const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                   const vector<int64_t>& blocksize, const float* bias, float* C,
                                   const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
  __m512 d = _mm512_setzero_ps();
#pragma omp parallel for collapse(2)
  for (int64_t mb = 0; mb < M / M_NBLK; mb++) {
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      for (int64_t i = 0; i < M_NBLK; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int64_t i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < M_NBLK; i++) {
        output[i] = _mm512_max_ps(output[i], d);
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_max_ps(output[i], d);
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
}

void sparse_gemm_bsc_bias_sum_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                  const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                  const vector<int64_t>& blocksize, const float* bias, const float* post, float* C,
                                  const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; mb++) {
    for (int b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      for (int i = 0; i < M_NBLK; i++) {
        output[i] =
            _mm512_add_ps(_mm512_load_ps(&bias[b_col * 16]), _mm512_load_ps(post + (mb * M_NBLK + i) * N + b_col * 16));
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int i = 0; i < M_NBLK; i++) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_add_ps(_mm512_load_ps(&bias[b_col * 16]),
                             _mm512_load_ps(post + (tail_row_bgn + i) * N + b_col * 16));
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
}

static inline void replace_tanh(float* activation, int N = 16) {
  for (int i = 0; i < N; i += 1) {
    float plus = expf(activation[i]);
    float minus = expf(-activation[i]);
    activation[i] = (plus - minus) / (plus + minus);
  }
}

void sparse_gemm_bsc_bias_tanh_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                   const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                   const vector<int64_t>& blocksize, const float* bias, float* C,
                                   const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; mb++) {
    for (int b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      for (int i = 0; i < M_NBLK; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int i = 0; i < M_NBLK; i++) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
#pragma omp parallel for
  for (int i = 0; i < M * N; i += 16) {
    replace_tanh(C + i);
  }
}

// gelu_tanh
static inline void i_gelu_tanh(float* activation, int len = 16) {
  for (int i = 0; i < len; ++i) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    float v = sqrt_2_over_pi * activation[i] * (1.f + fitting_const * activation[i] * activation[i]);
    // compute tanh(v)
    float tanh_v = 1.f;
    if (v < -7.5944) {
      tanh_v = -1.f;
    } else if (v < 7.5944) {
      float exp_v = expf(v);
      float r_exp_v = 1.f / exp_v;
      tanh_v = (exp_v - r_exp_v) / (exp_v + r_exp_v);
    }
    // compute gelu_tanh
    activation[i] = 0.5f * activation[i] * (1.f + tanh_v);
  }
}

void sparse_gemm_bsc_bias_gelu_tanh_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                        const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                        const vector<int64_t>& blocksize, const float* bias, float* C,
                                        const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; mb++) {
    for (int b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      // load bias
      output[0] = _mm512_load_ps(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; i++) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int i = 0; i < M_NBLK; ++i) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
// gelu_tanh
#pragma omp parallel for
  for (int i = 0; i < M * N; i += 16) {
    i_gelu_tanh(C + i);
  }
}

// sigmoid
static inline void i_sigmoid(float* activation, int len = 16) {
  for (int i = 0; i < len; ++i) {
    activation[i] = 1.f / (1.f + expf(-activation[i]));
  }
}

void sparse_gemm_bsc_bias_sigmod_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                     const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                     const vector<int64_t>& blocksize, const float* bias, float* C,
                                     const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; mb++) {
    for (int b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      __m512 output[M_NBLK];
      // load bias
      output[0] = _mm512_load_ps(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; i++) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512 activation[M_NBLK];
        for (int i = 0; i < M_NBLK; i++) {
          activation[i] = _mm512_set1_ps(A[(mb * M_NBLK + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int i = 0; i < M_NBLK; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int i = 0; i < M_NBLK; ++i) {
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, output[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int64_t b_col = 0; b_col < ncolptr - 1; b_col++) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512 output[kTailCnt];
      for (int64_t i = 0; i < tail_row_num; i++) {
        output[i] = _mm512_load_ps(&bias[b_col * 16]);
      }
      for (int64_t b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; b_row_idx++) {
        int64_t b_row = rowidxs[b_row_idx];
        __m512 activation[kTailCnt];
        for (int64_t i = 0; i < tail_row_num; i++) {
          activation[i] = _mm512_set1_ps(A[(tail_row_bgn + i) * K + b_row]);
        }
        __m512 sparse_weight = _mm512_load_ps(&B[b_row_idx * 16]);
        for (int64_t i = 0; i < tail_row_num; i++) {
          output[i] = _mm512_fmadd_ps(sparse_weight, activation[i], output[i]);
        }
      }
      for (int64_t i = 0; i < tail_row_num; i++) {
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, output[i]);
      }
    }
  }
  // sigmoid
#pragma omp parallel for
  for (int i = 0; i < M * N; i += 16) {
    i_sigmoid(C + i);
  }
}
#endif

#if __AVX512VNNI__
/********* int8 kernel *********/
// output f32
void sparse_gemm_bsc_4x16_u8s8f32(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                  const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                  const int* bias, const float scale, float* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();
  __m512 _scale = _mm512_set1_ps(scale);

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, c_f32[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, c_f32[i]);
      }
    }
  }
}

// Fuse ReLu, output f32
void sparse_gemm_bsc_4x16_u8s8f32_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                       const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                       const int* bias, const float scale, float* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();
  __m512 _scale = _mm512_set1_ps(scale);

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);
        _mm512_store_ps(C + (mb * M_NBLK + i) * N + b_col * 16, c_f32[i]);
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);
        _mm512_store_ps(C + (tail_row_bgn + i) * N + b_col * 16, c_f32[i]);
      }
    }
  }
}

// Fuse ReLu, output u8
void sparse_gemm_bsc_4x16_u8s8u8_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                      const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                      const int* bias, const float scale, uint8_t* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();
  __m512 _scale = _mm512_set1_ps(scale);

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);

        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        // if output is u8, we assume we will do ReLU,
        // so there's no negative values in output,
        // and we will use unsigned int later.
        output[i] = _mm512_cvt_roundps_epu32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (mb * M_NBLK + i) * N + b_col * 16, _mm512_cvtusepi32_epi8(output[i]));
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);

        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        // if output is u8, we assume we will do ReLU,
        // so there's no negative values in output,
        // and we will use unsigned int later.
        output[i] = _mm512_cvt_roundps_epu32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (tail_row_bgn + i) * N + b_col * 16, _mm512_cvtusepi32_epi8(output[i]));
      }
    }
  }
}

// Fuse ReLu, output u8, per channel
void sparse_gemm_bsc_4x16_u8s8u8_pc_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                         const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                         const int* bias, const vector<float>& scale, uint8_t* C,
                                         const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 _scale = _mm512_loadu_ps(&(scale[b_col << 4]));
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);

        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        // if output is u8, we assume we will do ReLU,
        // so there's no negative values in output,
        // and we will use unsigned int later.
        output[i] = _mm512_cvt_roundps_epu32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (mb * M_NBLK + i) * N + b_col * 16, _mm512_cvtusepi32_epi8(output[i]));
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 _scale = _mm512_loadu_ps(&(scale[b_col << 4]));
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        // ReLU
        output[i] = _mm512_max_epi32(output[i], zero);

        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        // if output is u8, we assume we will do ReLU,
        // so there's no negative values in output,
        // and we will use unsigned int later.
        output[i] = _mm512_cvt_roundps_epu32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (tail_row_bgn + i) * N + b_col * 16, _mm512_cvtusepi32_epi8(output[i]));
      }
    }
  }
}

// output s8
void sparse_gemm_bsc_4x16_u8s8s8(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                 const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                 const int* bias, const float scale, int8_t* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();
  __m512 _scale = _mm512_set1_ps(scale);

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        output[i] = _mm512_cvt_roundps_epi32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (mb * M_NBLK + i) * N + b_col * 16, _mm512_cvtsepi32_epi8(output[i]));
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        output[i] = _mm512_cvt_roundps_epi32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (tail_row_bgn + i) * N + b_col * 16, _mm512_cvtsepi32_epi8(output[i]));
      }
    }
  }
}

// output s8, per channel
void sparse_gemm_bsc_4x16_u8s8s8_pc(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                    const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                    const int* bias, const vector<float>& scale, int8_t* C, const int64_t M_NBLK) {
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);

  __m512i zero = _mm512_setzero_epi32();

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; ++mb) {
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      __m512i output[M_NBLK];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < M_NBLK; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[M_NBLK];
        for (int i = 0; i < M_NBLK; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (mb * M_NBLK + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < M_NBLK; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 _scale = _mm512_loadu_ps(&(scale[b_col << 4]));
      __m512 c_f32[M_NBLK];
      for (int i = 0; i < M_NBLK; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        output[i] = _mm512_cvt_roundps_epi32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (mb * M_NBLK + i) * N + b_col * 16, _mm512_cvtsepi32_epi8(output[i]));
      }
    }
  }
  int tail_row_bgn = M / M_NBLK * M_NBLK;
  int tail_row_num = M - tail_row_bgn;
  if (tail_row_num != 0) {
#pragma omp parallel for
    for (int b_col = 0; b_col < ncolptr - 1; ++b_col) {  // N dim
      const size_t kTailCnt = tail_row_num;
      __m512i output[kTailCnt];
      // load bias
      output[0] = _mm512_load_epi32(bias + b_col * 16);
      for (int i = 1; i < tail_row_num; ++i) {
        output[i] = output[0];
      }
      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col + 1]; ++b_row_idx) {  // K dim
        int b_row = rowidxs[b_row_idx];
        __m512i activation[kTailCnt];
        for (int i = 0; i < tail_row_num; ++i) {
          activation[i] = _mm512_set1_epi32(*reinterpret_cast<const int*>(A + (tail_row_bgn + i) * K + b_row));
        }
        __m512i sparse_weight = _mm512_load_epi32(&B[b_row_idx << 6]);
        for (int i = 0; i < tail_row_num; ++i) {
          output[i] = _mm512_dpbusds_epi32(output[i], activation[i], sparse_weight);
        }
      }
      __m512 _scale = _mm512_loadu_ps(&(scale[b_col << 4]));
      __m512 c_f32[kTailCnt];
      for (int i = 0; i < tail_row_num; ++i) {
        c_f32[i] = _mm512_cvt_roundepi32_ps(output[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        c_f32[i] = _mm512_mul_ps(c_f32[i], _scale);

        output[i] = _mm512_cvt_roundps_epi32(c_f32[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_store_epi64(C + (tail_row_bgn + i) * N + b_col * 16, _mm512_cvtsepi32_epi8(output[i]));
      }
    }
  }
}
#endif
}  // namespace executor
