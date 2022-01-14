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

#ifndef ENGINE_EXECUTOR_INCLUDE_SPARSE_OPERATORS_SPARSE_INNER_PRODUCT_HPP_
#define ENGINE_EXECUTOR_INCLUDE_SPARSE_OPERATORS_SPARSE_INNER_PRODUCT_HPP_
#include <assert.h>
#include <immintrin.h>
#include <math.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../common.hpp"

namespace executor {

template <typename T>
void TransposeMatrix(const T* input, const vector<int64_t>& shape, T* output);

template <typename T>
float GetSparseRatio(const T* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize);

// BSR definition follows scipy
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
// "data" is in (nnz, blocksize[0], blocksize[1])
// shape[0] -> blocksize[0], shape[1] -> blocksize[1]
// blocksize evenly divide corresponding shape
template <typename T>
struct BSRMatrix {
  vector<int64_t> shape;
  vector<int64_t> blocksize;
  int64_t nnz;
  int64_t nrowptr;
  T* data;
  int64_t* colidxs;
  int64_t* rowptr;
};

template <typename T>
struct BSCMatrix {
  vector<int64_t> shape;
  vector<int64_t> blocksize;
  int64_t nnz;
  int64_t ncolptr;
  T* data;
  int64_t* rowidxs;
  int64_t* colptr;
};

template <typename T>
BSRMatrix<T>* create_bsr_matrix(const T* dense_matrix, const vector<int64_t>& shape, const vector<int64_t>& blocksize);

template <typename T>
BSCMatrix<T>* create_bsc_matrix(const T* dense_matrix, const vector<int64_t>& shape, const vector<int64_t>& blocksize);

template <typename T>
void destroy_bsr_matrix(BSRMatrix<T>* bsr_matrix);

template <typename T>
void destroy_bsc_matrix(BSCMatrix<T>* bsc_matrix);

void reorder_bsc_int8_4x16(BSCMatrix<int8_t>* bsc);

#if __AVX512F__
/********* fp32 kernel *********/
void sparse_gemm_bsc_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B, const int64_t* rowidxs,
                         const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize, float* C,
                         const int64_t M_NBLK);

void sparse_gemm_bsc_bias_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B, const int64_t* rowidxs,
                              const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                              const float* bias, float* C, const int64_t M_NBLK);

void sparse_gemm_bsc_bias_relu_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                   const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                   const vector<int64_t>& blocksize, const float* bias, float* C, const int64_t M_NBLK);

void sparse_gemm_bsc_bias_sum_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                  const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                  const vector<int64_t>& blocksize, const float* bias,
                                  const float* post,  // append sum tensor data
                                  float* C, const int64_t M_NBLK);

void sparse_gemm_bsc_bias_tanh_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                   const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                   const vector<int64_t>& blocksize, const float* bias, float* C, const int64_t M_NBLK);

void sparse_gemm_bsc_bias_gelu_tanh_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                        const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                        const vector<int64_t>& blocksize, const float* bias, float* C,
                                        const int64_t M_NBLK);

void sparse_gemm_bsc_bias_sigmod_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* B,
                                     const int64_t* rowidxs, const int64_t* colptr, const int64_t ncolptr,
                                     const vector<int64_t>& blocksize, const float* bias, float* C,
                                     const int64_t M_NBLK);
#endif
#if __AVX512VNNI__
/********* int8 kernel *********/
// output f32
void sparse_gemm_bsc_4x16_u8s8f32(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                  const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                  const int* bias, const float scale, float* C, const int64_t M_NBLK);

// Fuse ReLu, output f32
void sparse_gemm_bsc_4x16_u8s8f32_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                       const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                       const int* bias, const float scale, float* C, const int64_t M_NBLK);

// Fuse ReLu, output u8
void sparse_gemm_bsc_4x16_u8s8u8_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                      const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                      const int* bias, const float scale, uint8_t* C, const int64_t M_NBLK);

// Fuse ReLu, output u8, per channel
void sparse_gemm_bsc_4x16_u8s8u8_pc_relu(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                         const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                         const int* bias, const vector<float>& scale, uint8_t* C, const int64_t M_NBLK);

// output s8
void sparse_gemm_bsc_4x16_u8s8s8(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                 const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                 const int* bias, const float scale, int8_t* C, const int64_t M_NBLK);

// output s8, per channel
void sparse_gemm_bsc_4x16_u8s8s8_pc(int M, int N, int K, const uint8_t* A, const int8_t* B, const int64_t* rowidxs,
                                    const int64_t* colptr, const int64_t ncolptr, const vector<int64_t>& blocksize,
                                    const int* bias, const vector<float>& scale, int8_t* C, const int64_t M_NBLK);
#endif
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_SPARSE_OPERATORS_SPARSE_INNER_PRODUCT_HPP_
