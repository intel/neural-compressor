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

#ifndef ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_
#define ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_

#include <float.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <limits.h>

#include <chrono>  // NOLINT
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <functional>
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>  // pair
#include <vector>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "memory_allocator.hpp"

#if __AVX512F__
#include <immintrin.h>
#endif

namespace executor {

using std::max;
using std::min;
using std::set;
using std::unordered_map;
using std::vector;
namespace ipc = boost::interprocess;

void GlobalInit(int* pargc, char*** pargv);

extern unordered_map<string, int> type2bytes;

// read weight file to data
void* read_file_to_type(const string& root, const string& type, const vector<int64_t>& shape,
                        const vector<int64_t>& location);

ipc::managed_shared_memory::handle_t load_shared_weight(const string& root, const string& type,
                                                        const vector<int64_t>& shape, const vector<int64_t>& location);

void InitVector(float* v, int buffer_size);

int64_t Product(const vector<int64_t>& shape);

// Get the shapes vector with the absolute perm. Default or empty perm is (0, 1, 2, 3, ...).
// e.g.: shape_before = (64, 384, 16, 64), perm = (0, 2, 1, 3), return (64, 16, 384, 64)
vector<int64_t> GetShapes(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm = {});

// Get the strides vector with the absolute perm. Default or empty perm is (0, 1, 2, 3, ...).
// Tensor stride is a product of its higher dimensions, Stride[0] = Shape[1]*Shape[2]*...*Shape[n].
// e.g.: axis = (0, 1, 2, 3), shape = (64, 16, 384, 64), return stride = (16*384*64, 384*64, 64, 1)
vector<int64_t> GetStrides(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm = {});

template <typename T>
T StringToNum(const string& str);

// Compare two buffer
template <typename T>
bool CompareData(const void* buf1, int64_t elem_num1, const void* buf2, int64_t elem_num2, float eps = 1e-6);

vector<float> GetScales(const void* mins, const void* maxs, const int64_t size, const string& dtype);

vector<float> GetRescales(const vector<float>& src0_scales, const vector<float>& src1_scales,
                          const vector<float>& dst_scales, const string& dst_dtype, const bool append_eltwise = false);

vector<int> GetZeroPoints(const void* mins, const vector<float>& scales, const string& dtype);

void AddZeroPoints(const int size, const string& dtype, const float* src_data, const float* range_mins,
                   const vector<float>& scales, float* dst_data);

#if __AVX512F__
void Quantize_avx512(const int size, const string& dtype, const void* src_data, const float* range_mins,
                     const vector<float>& scales, void* dst_data);
#else
void Quantize(const int size, const string& dtype, const void* src_data, const float* range_mins,
              const vector<float>& scales, void* dst_data);
#endif

vector<int64_t> ReversePerm(const vector<int64_t>& perm_to);

float Time(string state);

template <typename T>
void PrintToFile(const T* data, const std::string& name, size_t size = 1000);

template <typename T>
void StringSplit(vector<T>* split_list, const string& str_list, const string& split_op);

void InitSparse(int K, int N, int N_BLKSIZE, int K_BLKSIZE, int N_SPARSE, float* A);

/************* ref ************/
template <typename dst_type, typename src_type>
void ref_mov_ker(dst_type* inout, const src_type* in, size_t len);
template <typename dst_type, typename src_type>
void ref_add_ker(dst_type* inout, src_type* in, size_t len);
/************* fp32 ************/
void zero_ker(float* out, size_t len);
void move_ker(float* out, const float* in, size_t len);
void add_ker(float* inout, float* in, size_t len);
/************* bf16 ************/
#if __AVX512F__
// Conversion from BF16 to FP32
__m512 cvt_bf16_to_fp32(const __m256i src);
// Conversion from FP32 to BF16
__m256i trunc_fp32_to_bf16(const __m512 src);
__m256i cvt_fp32_to_bf16(const __m512 src);
#endif
void zero_ker(uint16_t* out, size_t len);
void move_ker(uint16_t* out, const uint16_t* in, size_t len);
void add_ker(uint16_t* inout, uint16_t* in, size_t len);
/************* int8 ************/
void zero_ker(uint8_t* out, size_t len);
void move_ker(uint8_t* out, const uint8_t* in, size_t len);
void add_ker(uint8_t* inout, uint8_t* in, size_t len);

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_
