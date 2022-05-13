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
#include "common.hpp"

#include "cmath"

namespace executor {

unordered_map<string, int> type2bytes = {{"fp32", sizeof(float)},       {"int8", sizeof(char)}, {"int32", sizeof(int)},
                                         {"u8", sizeof(unsigned char)}, {"s8", sizeof(char)},   {"s32", sizeof(int)},
                                         {"bf16", sizeof(uint16_t)}};
const int CPU_COUNT = omp_get_max_threads();
#if __AVX512F__
const int ALIGN_NUM = 64;
#else
const int ALIGN_NUM = 32;
#endif

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  FLAGS_logtostderr = 1;
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}
// read weight file to data
/*
Args:
    root is the model const tensors like weights, bias .bin file path.
    type is for setting the tensor datatype, like 'float32'
    shape is the data shape
    location gives the info of tensor data location in .bin file
      location[0] is the start idx when sotre the data bytes
      location[1] is the data bytes length
Return:
    void* ptr, points a consecutive memory that sotres the data
*/
void* read_file_to_type(const string& root, const string& type, const vector<int64_t>& shape,
                        const vector<int64_t>& location) {
  int b = type2bytes[type];
  if (b == 0) {
    LOG(INFO) << type << " not implemented yet...";
  }

  int64_t size = Product(shape);
  // from file tensor will directly malloc memory
  void* p = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size * b / ALIGNMENT + 1) * ALIGNMENT));

  std::ifstream inFile(root, std::ios::in | std::ios::binary);
  if (inFile) {
    inFile.seekg(location[0], std::ios::beg);
    inFile.read(reinterpret_cast<char*>(p), location[1]);
    inFile.close();
  } else {
    std::memcpy(p, &root[location[0]], location[1]);
  }
  return p;
}

void InitVector(float* v, int buffer_size) {
  std::mt19937 gen;
  static int seed = 0;
  gen.seed(seed);
  std::uniform_real_distribution<float> u(-10, 10);
  for (int i = 0; i < buffer_size; ++i) {
    v[i] = u(gen);
  }
  seed++;
}

float Time(string state) {
  static std::chrono::microseconds micros_start = std::chrono::microseconds();
  static std::chrono::microseconds micros_end = std::chrono::microseconds();
  std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
  std::chrono::nanoseconds nanos = now_time.time_since_epoch();
  if (state == "start") {
    micros_start = std::chrono::duration_cast<std::chrono::microseconds>(nanos);
    return 0.;
  } else if (state == "end") {
    micros_end = std::chrono::duration_cast<std::chrono::microseconds>(nanos);
    return (micros_end.count() - micros_start.count()) / 1000.0;
  } else {
    LOG(FATAL) << "not supported state for time, only start and end...";
    return 0;
  }
}

int64_t Product(const vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
}

// Get the shapes vector with the absolute perm. Default or empty perm is (0, 1,
// 2, 3, ...). e.g.: shape_before = (64, 384, 16, 64), perm = (0, 2, 1, 3),
// return (64, 16, 384, 64)
vector<int64_t> GetShapes(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm) {
  if (absolute_perm.empty()) {
    return origin_shape;
  }
  int shape_size = origin_shape.size();
  vector<int64_t> transed_shape(shape_size, 0);
#pragma omp parallel for
  for (int i = 0; i < shape_size; ++i) {
    int trans_axis_id = absolute_perm[i];
    transed_shape[i] = origin_shape[trans_axis_id];
  }
  return transed_shape;
}

// Get the strides vector with the absolute perm. Default or empty perm is (0,
// 1, 2, 3, ...). Tensor's each stride is the product of all higher dimensions
// Stride[0] = Shape(1)*Shape(2)*...*Shape(n).
// e.g.: axis = (0, 1, 2, 3), shape = (64, 16, 384, 64), return stride =
// (16*384*64, 384*64, 64, 1)
vector<int64_t> GetStrides(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm) {
  int shape_size = origin_shape.size();
  vector<int64_t> origin_strides(shape_size, 1);
  if (shape_size >= 2) {
    for (int i = shape_size - 2; i >= 0; --i) {
      origin_strides[i] = origin_shape[i + 1] * origin_strides[i + 1];
    }
  }
  return GetShapes(origin_strides, absolute_perm);
}

template <typename T>
bool CompareData(const void* buf1, int64_t elem_num1, const void* buf2, int64_t elem_num2, float eps) {
  if (buf1 == buf2) {
    return false;
  }
  if (elem_num1 != elem_num2) {
    return false;
  }
  const auto buf1_data = static_cast<const T*>(buf1);
  const auto buf2_data = static_cast<const T*>(buf2);
  for (int64_t i = 0; i < elem_num1; ++i) {
    auto err = fabs(buf1_data[i] - buf2_data[i]);
    if (err > eps) {
      return false;
    }
  }
  return true;
}
template bool CompareData<float>(const void* buf1, int64_t elem_num1, const void* buf2, int64_t elem_num2, float eps);

vector<float> GetScales(const void* mins, const void* maxs, const int64_t size, const string& dtype) {
  const float* mins_p = static_cast<const float*>(mins);
  const float* maxs_p = static_cast<const float*>(maxs);

  vector<float> scales;
  if (dtype == "u8") {
    for (int i = 0; i < size; i++) {
      float max_sub_min = maxs_p[i] - mins_p[i];
      max_sub_min = max_sub_min < 1e-10 ? 1e-10 : max_sub_min;
      scales.emplace_back(255.f / max_sub_min);
    }
  } else if (dtype == "s8") {
    for (int i = 0; i < size; i++) {
      float abs_max = max(abs(maxs_p[i]), abs(mins_p[i]));
      abs_max = abs_max < 1e-10 ? 1e-10 : abs_max;
      scales.emplace_back(127.f / abs_max);
    }
  } else if (dtype == "fp32") {
    for (int i = 0; i < size; i++) {
      scales.emplace_back(1.f);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return scales;
}

vector<float> GetRescales(const vector<float>& src0_scales, const vector<float>& src1_scales,
                          const vector<float>& dst_scales, const string& dst_dtype, const bool append_eltwise) {
  vector<float> rescales;
  if (dst_dtype == "fp32") {
    for (int i = 0; i < src1_scales.size(); i++) {
      rescales.emplace_back(1. / (src0_scales[0] * src1_scales[i]));
    }
  } else if (dst_dtype == "s8" && !dst_scales.empty()) {
    for (int i = 0; i < src1_scales.size(); i++) {
      auto rescale =
          append_eltwise ? 1. / (src0_scales[0] * src1_scales[i]) : dst_scales[0] / (src0_scales[0] * src1_scales[i]);
      rescales.emplace_back(rescale);
    }
  } else if (dst_dtype == "u8" && !dst_scales.empty()) {
    for (int i = 0; i < src1_scales.size(); i++) {
      auto rescale =
          append_eltwise ? 1. / (src0_scales[0] * src1_scales[i]) : dst_scales[0] / (src0_scales[0] * src1_scales[i]);
      rescales.emplace_back(rescale);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dst_dtype << " now!";
  }
  return rescales;
}

vector<int> GetZeroPoints(const void* mins, const vector<float>& scales, const string& dtype) {
  const float* mins_p = static_cast<const float*>(mins);
  vector<int> zerops;
  if (dtype == "u8") {
    for (int i = 0; i < scales.size(); i++) zerops.emplace_back(nearbyint(-mins_p[i] * scales[i]));
  } else if (dtype == "s8") {
    for (int i = 0; i < scales.size(); i++) zerops.emplace_back(nearbyint(-128 - mins_p[i] * scales[i]));
  } else {
    LOG(ERROR) << "Can't suppport dtype: " << dtype << " now!";
  }
  return zerops;
}

void AddZeroPoints(const int size, const string& dtype, const float* src_data, const float* range_mins,
                   const vector<float>& scales, float* dst_data) {
  if (dtype == "u8") {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      dst_data[i] = src_data[i] - range_mins[0];
    }
  } else if (dtype == "s8") {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      dst_data[i] = src_data[i] - 128 / scales[0] - range_mins[0];
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return;
}

#if __AVX512F__
void Quantize_avx512(const int size, const string& dtype, const void* src_data, const float* range_mins,
                     const vector<float>& scales, void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);

  int avx512_loop_len = size >> 4;

  if (dtype == "bf16") {
    uint16_t* dst_data_ = static_cast<uint16_t*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < avx512_loop_len; ++i) {
      __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
#if __AVX512_BF16__
      __m256i data_bf16 = (__m256i)_mm512_cvtneps_pbh(_src_data);
#else
      auto y = _mm512_bsrli_epi128(_mm512_castps_si512(_src_data), 2);
      __m256i data_bf16 = _mm512_cvtepi32_epi16(y);
#endif
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_data_ + (i << 4)), data_bf16);
    }
    union {
      unsigned int u;
      float f;
    } typecast;
#pragma omp parallel for
    for (int i = (avx512_loop_len << 4); i < size; i++) {
      typecast.f = src_data_[i];
      dst_data_[i] = typecast.u >> 16;
    }
    return;
  }

  __m512 _min_with_scale_u8 = _mm512_set1_ps(range_mins[0] * scales[0]);
  __m512 _min_with_scale_s8 = _mm512_set1_ps(0);
  __m512 _scale = _mm512_set1_ps(scales[0]);
  __m512i zero = _mm512_setzero_epi32();

  if (dtype == "u8") {
    unsigned char* dst_data_ = static_cast<unsigned char*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < avx512_loop_len; ++i) {
      __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
      __m512 data = _mm512_fmsub_ps(_src_data, _scale, _min_with_scale_u8);
      __m512i data_x32 = _mm512_cvt_roundps_epi32(data, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      data_x32 = _mm512_max_epi32(data_x32, zero);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + (i << 4)), _mm512_cvtusepi32_epi8(data_x32));
    }

#pragma omp parallel for
    for (int i = (avx512_loop_len << 4); i < size; i++) {
      int32_t data = nearbyint((src_data_[i] - range_mins[0]) * scales[0]);
      data = data < 0 ? 0 : data;
      data = data > 255 ? 255 : data;
      dst_data_[i] = static_cast<unsigned char>(data);
    }
  } else if (dtype == "s8") {
    char* dst_data_ = static_cast<char*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < avx512_loop_len; ++i) {
      __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
      __m512 data = _mm512_fmsub_ps(_src_data, _scale, _min_with_scale_s8);
      __m512i data_x32 = _mm512_cvt_roundps_epi32(data, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + (i << 4)), _mm512_cvtsepi32_epi8(data_x32));
    }
#pragma omp parallel for
    for (int i = (avx512_loop_len << 4); i < size; i++) {
      int32_t data = nearbyint(src_data_[i] * scales[0]);
      data = data < -128 ? -128 : data;
      data = data > 127 ? 127 : data;
      dst_data_[i] = static_cast<char>(data);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return;
}

#else
void Quantize(const int size, const string& dtype, const void* src_data, const float* range_mins,
              const vector<float>& scales, void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);
  if (dtype == "u8") {
    unsigned char* dst_data_ = static_cast<unsigned char*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      int32_t data = nearbyint((src_data_[i] - range_mins[0]) * scales[0]);
      data = data < 0 ? 0 : data;
      data = data > 255 ? 255 : data;
      dst_data_[i] = static_cast<unsigned char>(data);
    }
  } else if (dtype == "s8") {
    char* dst_data_ = static_cast<char*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      int32_t data = nearbyint(src_data_[i] * scales[0]);
      data = data < -128 ? -128 : data;
      data = data > 127 ? 127 : data;
      dst_data_[i] = static_cast<char>(data);
    }
  } else if (dtype == "bf16") {
    uint16_t* dst_data_ = static_cast<uint16_t*>(dst_data);
    union {
      unsigned int u;
      float f;
    } typecast;
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      typecast.f = src_data_[i];
      dst_data_[i] = typecast.u >> 16;
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return;
}
#endif

// Transpose from A to default B.
// e.g.: transpose from {2, 0, 1} to default {0, 1, 2} is {1, 2, 0}
vector<int64_t> ReversePerm(const vector<int64_t>& perm_to) {
  if (perm_to.empty()) return {};
  int dsize = perm_to.size();
  vector<int64_t> perm_from(dsize, 0);
  std::iota(perm_from.begin(), perm_from.end(), 0);
  if (perm_to.empty()) {
    return perm_from;
  }
#pragma omp parallel for
  for (int i = 0; i < dsize; ++i) {
    int index = perm_to[i];
    perm_from[index] = i;
  }
  return perm_from;
}

template <typename T>
T StringToNum(const string& str) {
  std::istringstream iss(str);
  T num;
  iss >> num;
  return num;
}

template float StringToNum<float>(const string& str);
template int64_t StringToNum<int64_t>(const string& str);

template <typename T>
void PrintToFile(const T* data, const std::string& name, size_t size) {
  // print output file
  auto pos = name.rfind("/");
  string output_file = (pos != string::npos ? name.substr(pos + 1) : name) + ".txt";
  std::ofstream output_data(output_file);
  for (size_t i = 0; i < size; ++i) {
    output_data << static_cast<float>(data[i]) << "\n";
  }
  output_data.close();
}
template void PrintToFile<float>(const float* data, const std::string& name, size_t size);
template void PrintToFile<unsigned char>(const unsigned char* data, const std::string& name, size_t size);
template void PrintToFile<char>(const char* data, const std::string& name, size_t size);
template void PrintToFile<int32_t>(const int32_t* data, const std::string& name, size_t size);
template void PrintToFile<int64_t>(const int64_t* data, const std::string& name, size_t size);

template <typename T>
void StringSplit(vector<T>* split_list, const string& str_list, const string& split_op) {
  std::string::size_type pos1 = 0;
  std::string::size_type pos2 = str_list.find(split_op);
  while (std::string::npos != pos2) {
    T element = StringToNum<T>(str_list.substr(pos1, pos2));
    split_list->push_back(element);
    pos1 = pos2 + split_op.size();
    pos2 = str_list.find(split_op, pos1);
  }
  if (pos1 != str_list.size()) {
    T element = StringToNum<T>(str_list.substr(pos1));
    split_list->push_back(element);
  }
}
template void StringSplit<float>(vector<float>* split_list, const string& string_list, const string& split_op);
template void StringSplit<int64_t>(vector<int64_t>* split_list, const string& string_list, const string& split_op);
template void StringSplit<char>(vector<char>* split_list, const string& string_list, const string& split_op);
template void StringSplit<unsigned char>(vector<unsigned char>* split_list, const string& string_list,
                                         const string& split_op);

void InitSparse(int K, int N, int N_BLKSIZE, int K_BLKSIZE, int N_SPARSE, float* B) {
  unsigned int seed = 0;
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      B[k * N + n] = rand_r(&seed) % 11 - 5;
    }
  }
  // sparsify B
  for (int nb = 0; nb < N / N_BLKSIZE; nb++) {
    for (int kb = 0; kb < K / K_BLKSIZE; kb++) {
      bool zero_fill = rand_r(&seed) % N_SPARSE != 0;
      if (zero_fill) {
        for (int n = 0; n < N_BLKSIZE; n++) {
          for (int k = 0; k < K_BLKSIZE; k++) {
            B[(kb * K_BLKSIZE + k) * N + nb * N_BLKSIZE + n] = 0;
          }
        }
      }
    }
  }
}

/************* ref ************/
template <typename dst_type, typename src_type>
void ref_mov_ker(dst_type* inout, const src_type* in, size_t len) {
#pragma omp palleral for
  for (int i = 0; i < len; i++) {
    *(inout + i) = *(in + i);
  }
}
template void ref_mov_ker(float* inout, const float* in, size_t len);
template void ref_mov_ker(uint16_t* inout, const uint16_t* in, size_t len);
template void ref_mov_ker(uint8_t* inout, const uint8_t* in, size_t len);

template <typename dst_type, typename src_type>
void ref_add_ker(dst_type* inout, src_type* in, size_t len) {
#pragma omp palleral for
  for (int i = 0; i < len; i++) {
    *(inout + i) += *(in + i);
  }
}
template void ref_add_ker(float* inout, float* in, size_t len);
template void ref_add_ker(uint16_t* inout, uint16_t* in, size_t len);
template void ref_add_ker(uint8_t* inout, uint8_t* in, size_t len);

/************* fp32 ************/
void zero_ker(float* out, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  __m512 zero_512 = _mm512_setzero_ps();
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_ps(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_ps(out + i, mask, zero_512);
  }
#else
  memset(out, 0, len * sizeof(float));
#endif
}

void move_ker(float* out, const float* in, size_t len) {
  int64_t i = 0;
#if __AVX512F__
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(float* inout, float* in, size_t len) {
  int i = 0;
#if __AVX512F__
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto out2 = _mm512_loadu_ps(inout + i + 16);
    auto in1 = _mm512_loadu_ps(in + i);
    auto in2 = _mm512_loadu_ps(in + i + 16);
    out1 = _mm512_add_ps(out1, in1);
    out2 = _mm512_add_ps(out2, in2);
    _mm512_storeu_ps(inout + i, out1);
    _mm512_storeu_ps(inout + i + 16, out2);
  }

  if (i < len - 15) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto in1 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(inout + i, _mm512_add_ps(out1, in1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto out1 = _mm512_maskz_loadu_ps(mask, inout + i);
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(inout + i, mask, _mm512_add_ps(out1, in1));
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

/************* bf16 ************/
#if __AVX512F__
// Conversion from BF16 to FP32
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}
// Conversion from FP32 to BF16
inline __m256i trunc_fp32_to_bf16(const __m512 src) {
  auto y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if __AVX512_BF16__
  return _mm512_cvtneps_pbh(src);
#else
  return trunc_fp32_to_bf16(src);
#endif
}
#endif

void zero_ker(uint16_t* out, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  __m512i zero_512 = _mm512_setzero_si512();
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi16(out + i, mask, zero_512);
  }
#else
  memset(out, 0, len * sizeof(uint16_t));
#endif
}

void move_ker(uint16_t* out, const uint16_t* in, size_t len) {
  int64_t i = 0;
#if __AVX512F__
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(uint16_t* inout, uint16_t* in, size_t len) {
  int i = 0;
#if __AVX512F__
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i)));
    auto inout2 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i + 16)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i), cvt_fp32_to_bf16(inout1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i + 16), cvt_fp32_to_bf16(inout2));
  }

  if (i < len - 15) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i), cvt_fp32_to_bf16(inout1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

/************* uint8 ************/
void zero_ker(uint8_t* out, size_t len) {
  int64_t i;
#if __AVX512F__
  __m512i zero_512 = _mm512_setzero_si512();
#pragma unroll(4)
  for (i = 0; i < len - 63; i += 64) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi8(out + i, mask, zero_512);
  }
#else
  memset(out, 0, sizeof(uint8_t) * len);
#endif
}

void move_ker(uint8_t* out, const uint8_t* in, size_t len) {
  int64_t i;
#if __AVX512F__
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(uint8_t* inout, uint8_t* in, size_t len) {
  int64_t i;
#if __AVX512F__
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    auto out = _mm512_loadu_si512(inout + i);
    out = _mm512_adds_epi8(out, in0);  // add with saturate
    _mm512_storeu_si512(inout + i, out);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    auto out = _mm512_maskz_loadu_epi8(mask, inout + i);
    out = _mm512_adds_epi8(out, in0);
    _mm512_mask_storeu_epi8(inout + i, mask, out);
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

void runtime_minmax(float* data, size_t length, float* min_num, float* max_num) {
  int block_size = (length / CPU_COUNT) / ALIGN_NUM * ALIGN_NUM;
  if (block_size == 0) {
    *min_num = *std::min_element(data, &data[length]);
    *max_num = *std::max_element(data, &data[length]);
    return;
  }
  int block_num = length / block_size;
  vector<float> block_mins(block_num + (length % block_size != 0)), block_maxs(block_num + (length % block_size != 0));
#if __AVX512F__
#pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    block_minmax_avx512(data + i * block_size, block_size, &block_mins[i], &block_maxs[i]);
  }
  if (length % block_size != 0) {
    block_minmax_avx512(data + block_num * block_size, length - block_num * block_size, &block_mins[block_num],
                        &block_maxs[block_num]);
  }
#else
#pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    block_minmax(data + i * block_size, block_size, &block_mins[i], &block_maxs[i]);
  }
  if (length % block_size != 0) {
    block_minmax(data + block_num * block_size, length - block_num * block_size, &block_mins[block_num],
                 &block_maxs[block_num]);
  }
#endif
  *min_num = *std::min_element(block_mins.begin(), block_mins.end());
  *max_num = *std::max_element(block_maxs.begin(), block_maxs.end());
}

void block_minmax_avx512(float* Input, size_t N, float* Min, float* Max) {
  float tmp_min = std::numeric_limits<float>::max();
  float tmp_max = std::numeric_limits<float>::lowest();

  if (N >= 16) {
    __m512 MaximumVector0 = _mm512_set1_ps(tmp_max);
    __m512 MinimumVector0 = _mm512_set1_ps(tmp_min);

    if (N >= 64) {
      __m512 MaximumVector1 = MaximumVector0;
      __m512 MaximumVector2 = MaximumVector0;
      __m512 MaximumVector3 = MaximumVector0;

      __m512 MinimumVector1 = MinimumVector0;
      __m512 MinimumVector2 = MinimumVector0;
      __m512 MinimumVector3 = MinimumVector0;

      while (N >= 64) {
        __m512 InputVector0 = _mm512_loadu_ps(Input);
        __m512 InputVector1 = _mm512_loadu_ps(Input + 16);
        __m512 InputVector2 = _mm512_loadu_ps(Input + 32);
        __m512 InputVector3 = _mm512_loadu_ps(Input + 48);

        MaximumVector0 = _mm512_max_ps(MaximumVector0, InputVector0);
        MaximumVector1 = _mm512_max_ps(MaximumVector1, InputVector1);
        MaximumVector2 = _mm512_max_ps(MaximumVector2, InputVector2);
        MaximumVector3 = _mm512_max_ps(MaximumVector3, InputVector3);

        MinimumVector0 = _mm512_min_ps(MinimumVector0, InputVector0);
        MinimumVector1 = _mm512_min_ps(MinimumVector1, InputVector1);
        MinimumVector2 = _mm512_min_ps(MinimumVector2, InputVector2);
        MinimumVector3 = _mm512_min_ps(MinimumVector3, InputVector3);

        Input += 64;
        N -= 64;
      }

      MaximumVector0 = _mm512_max_ps(MaximumVector0, MaximumVector1);
      MaximumVector2 = _mm512_max_ps(MaximumVector2, MaximumVector3);
      MaximumVector0 = _mm512_max_ps(MaximumVector0, MaximumVector2);

      MinimumVector0 = _mm512_min_ps(MinimumVector0, MinimumVector1);
      MinimumVector2 = _mm512_min_ps(MinimumVector2, MinimumVector3);
      MinimumVector0 = _mm512_min_ps(MinimumVector0, MinimumVector2);
    }

    while (N >= 16) {
      __m512 InputVector0 = _mm512_loadu_ps(Input);
      MaximumVector0 = _mm512_max_ps(MaximumVector0, InputVector0);
      MinimumVector0 = _mm512_min_ps(MinimumVector0, InputVector0);

      Input += 16;
      N -= 16;
    }

    float minx16[32];
    void* min_ptr = reinterpret_cast<void*>(minx16);
    std::size_t min_space = sizeof(minx16) - 1;
    std::align(64, sizeof(float), min_ptr, min_space);
    float* aligned_min = reinterpret_cast<float*>(min_ptr);
    _mm512_store_ps(aligned_min, MinimumVector0);
    float maxx16[32];
    void* max_ptr = reinterpret_cast<void*>(maxx16);
    std::size_t max_space = sizeof(maxx16) - 1;
    std::align(64, sizeof(float), max_ptr, max_space);
    float* aligned_max = reinterpret_cast<float*>(max_ptr);
    _mm512_store_ps(aligned_max, MaximumVector0);
    for (int i = 0; i < 16; i++) {
      tmp_max = std::max(tmp_max, aligned_max[i]);
      tmp_min = std::min(tmp_min, aligned_min[i]);
    }
  }

  while (N > 0) {
    tmp_max = std::max(tmp_max, *Input);
    tmp_min = std::min(tmp_min, *Input);

    Input += 1;
    N -= 1;
  }

  *Min = tmp_min;
  *Max = tmp_max;
}
void block_minmax(float* Input, size_t N, float* Min, float* Max) {
  float tmp_min = std::numeric_limits<float>::max();
  float tmp_max = std::numeric_limits<float>::lowest();

  if (N >= 8) {
    __m256 MaximumVector0 = _mm256_set1_ps(tmp_max);
    __m256 MinimumVector0 = _mm256_set1_ps(tmp_min);

    if (N >= 32) {
      __m256 MaximumVector1 = MaximumVector0;
      __m256 MaximumVector2 = MaximumVector0;
      __m256 MaximumVector3 = MaximumVector0;

      __m256 MinimumVector1 = MinimumVector0;
      __m256 MinimumVector2 = MinimumVector0;
      __m256 MinimumVector3 = MinimumVector0;

      while (N >= 32) {
        __m256 InputVector0 = _mm256_loadu_ps(Input);
        __m256 InputVector1 = _mm256_loadu_ps(Input + 8);
        __m256 InputVector2 = _mm256_loadu_ps(Input + 16);
        __m256 InputVector3 = _mm256_loadu_ps(Input + 24);

        MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
        MaximumVector1 = _mm256_max_ps(MaximumVector1, InputVector1);
        MaximumVector2 = _mm256_max_ps(MaximumVector2, InputVector2);
        MaximumVector3 = _mm256_max_ps(MaximumVector3, InputVector3);

        MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);
        MinimumVector1 = _mm256_min_ps(MinimumVector1, InputVector1);
        MinimumVector2 = _mm256_min_ps(MinimumVector2, InputVector2);
        MinimumVector3 = _mm256_min_ps(MinimumVector3, InputVector3);

        Input += 32;
        N -= 32;
      }

      MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector1);
      MaximumVector2 = _mm256_max_ps(MaximumVector2, MaximumVector3);
      MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector2);

      MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector1);
      MinimumVector2 = _mm256_min_ps(MinimumVector2, MinimumVector3);
      MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector2);
    }

    while (N >= 8) {
      __m256 InputVector0 = _mm256_loadu_ps(Input);
      MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
      MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);

      Input += 8;
      N -= 8;
    }

    float minx8[16];
    void* min_ptr = reinterpret_cast<void*>(minx8);
    std::size_t min_space = sizeof(minx8) - 1;
    std::align(32, sizeof(float), min_ptr, min_space);
    float* aligned_min = reinterpret_cast<float*>(min_ptr);
    _mm256_store_ps(aligned_min, MinimumVector0);
    float maxx8[16];
    void* max_ptr = reinterpret_cast<void*>(maxx8);
    std::size_t max_space = sizeof(maxx8) - 1;
    std::align(32, sizeof(float), max_ptr, max_space);
    float* aligned_max = reinterpret_cast<float*>(max_ptr);
    _mm256_store_ps(aligned_max, MaximumVector0);
    for (int i = 0; i < 8; i++) {
      tmp_max = std::max(tmp_max, aligned_max[i]);
      tmp_min = std::min(tmp_min, aligned_min[i]);
    }
  }

  while (N > 0) {
    tmp_max = std::max(tmp_max, *Input);
    tmp_min = std::min(tmp_min, *Input);

    Input += 1;
    N -= 1;
  }

  *Min = tmp_min;
  *Max = tmp_max;
}

/************ hash funtion for primitive cache ************/
template <typename T>
size_t hash_combine(size_t seed, const T &v) {
  return seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
void hash_val(size_t seed, const T& val) {
  hash_combine(seed, val);
}

template <typename T, typename... Types>
void hash_val(size_t seed, const T& val, const Types&... args) {
  hash_combine(seed, val);
  hash_val(seed, args...);
}

template <typename... Types>
size_t hash_val(const Types&... args) {
  size_t seed = 0;
  hash_val(seed, args...);
  return seed;
}

template <typename T>
size_t get_array_hash(size_t seed, const T& v, int size) {
  for (int i = 0; i < size; i++) {
    seed = hash_combine(seed, v[i]);
  }
  return seed;
}

/************ InnerProductPrimitiveFwdFactory member function ************/
size_t InnerProductPrimitiveFwdFactory::GenKey(const string& src0_dtype,
  const string& src1_dtype, const string& dst_dtype, const vector<int64_t>& src0_shape,
  const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm, const string& append_op,
  const vector<int64_t>& post_op_shape, const float& output_scale, const dnnl::engine* eng) {
  size_t seed = 0;
  // primitive kind
  string prefix = "inner_product_fwd_";
  seed = hash_val(prefix, src0_dtype, src1_dtype, dst_dtype);
  seed = get_array_hash(seed, src0_shape, src0_shape.size());
  seed = get_array_hash(seed, src1_shape, src1_shape.size());
  // if dst_shape has reverse_perm
  if (!dst_perm.empty()) {
    seed = get_array_hash(seed, dst_perm, dst_perm.size());
  }
  if (append_op != "") {
    seed = hash_combine(seed, append_op);
    if (append_op == "sum" || append_op == "binary_add") {
      seed = get_array_hash(seed, post_op_shape, post_op_shape.size());
    }
  }
  // if has output_scale
  if (output_scale != 1.f) {
    seed = hash_combine(seed, output_scale);
  }
  // hash each dnnl engine ptr
  seed = hash_combine(seed, eng);
  return seed;
}

size_t InnerProductPrimitiveFwdFactory::Key(const string& src0_dtype,
  const string& src1_dtype, const string& dst_dtype, const vector<int64_t>& src0_shape,
  const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm, const string& append_op,
  const vector<int64_t>& post_op_shape, const float& output_scale, const dnnl::engine* eng) {
  return InnerProductPrimitiveFwdFactory::GetInstance().GenKey(src0_dtype, src1_dtype, dst_dtype,
    src0_shape, src1_shape, dst_perm, append_op, post_op_shape, output_scale, eng);
}

bool InnerProductPrimitiveFwdFactory::IsInFactory(const size_t& key) {
  return InnerProductPrimitiveFwdFactory::GetInstance().IsInCache(key);
}

dnnl::inner_product_forward& InnerProductPrimitiveFwdFactory::Get(const size_t& key) {
  return static_cast<dnnl::inner_product_forward&>(
    InnerProductPrimitiveFwdFactory::GetInstance().GetPrimitive(key));
}

void InnerProductPrimitiveFwdFactory::Set(const size_t& key, dnnl::primitive primitive) {
  InnerProductPrimitiveFwdFactory::GetInstance().SetPrimitive(key, primitive);
}

void InnerProductPrimitiveFwdFactory::ClearFactory() {
  InnerProductPrimitiveFwdFactory::GetInstance().Clear();
}

bool InnerProductPrimitiveFwdFactory::DoNotCache() {
  return InnerProductPrimitiveFwdFactory::GetInstance().do_not_cache_;
}

InnerProductPrimitiveFwdFactory& InnerProductPrimitiveFwdFactory::GetInstance() {
  static InnerProductPrimitiveFwdFactory instance_;
  return instance_;
}

/************ MatMulPrimitiveFwdFactory member function ************/
size_t MatMulPrimitiveFwdFactory::GenKey(const string& src0_dtype,
  const string& src1_dtype, const string& dst_dtype, const vector<int64_t>& src0_shape,
  const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm, const string& append_op,
  const vector<int64_t>& post_op_shape, const float& output_scale, const dnnl::engine* eng) {
  size_t seed = 0;
  // primitive kind
  string prefix = "matmul_fwd_";
  seed = hash_val(prefix, src0_dtype, src1_dtype, dst_dtype);
  seed = get_array_hash(seed, src0_shape, src0_shape.size());
  seed = get_array_hash(seed, src1_shape, src1_shape.size());
  // if dst_shape has reverse_perm
  if (!dst_perm.empty()) {
    seed = get_array_hash(seed, dst_perm, dst_perm.size());
  }
  if (append_op != "") {
    seed = hash_combine(seed, append_op);
    if (append_op == "sum" || append_op == "binary_add") {
      seed = get_array_hash(seed, post_op_shape, post_op_shape.size());
    }
  }
  // if has output_scale
  if (output_scale != 1.f) {
    seed = hash_combine(seed, output_scale);
  }
  // hash each dnnl engine ptr
  seed = hash_combine(seed, eng);
  return seed;
}

size_t MatMulPrimitiveFwdFactory::Key(const string& src0_dtype,
  const string& src1_dtype, const string& dst_dtype, const vector<int64_t>& src0_shape,
  const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm, const string& append_op,
  const vector<int64_t>& post_op_shape, const float& output_scale, const dnnl::engine* eng) {
  return MatMulPrimitiveFwdFactory::GetInstance().GenKey(src0_dtype, src1_dtype, dst_dtype,
    src0_shape, src1_shape, dst_perm, append_op, post_op_shape, output_scale, eng);
}

bool MatMulPrimitiveFwdFactory::IsInFactory(const size_t& key) {
  return MatMulPrimitiveFwdFactory::GetInstance().IsInCache(key);
}

dnnl::matmul& MatMulPrimitiveFwdFactory::Get(const size_t& key) {
  return static_cast<dnnl::matmul&>(
    MatMulPrimitiveFwdFactory::GetInstance().GetPrimitive(key));
}

void MatMulPrimitiveFwdFactory::Set(const size_t& key, dnnl::primitive primitive) {
  MatMulPrimitiveFwdFactory::GetInstance().SetPrimitive(key, primitive);
}

void MatMulPrimitiveFwdFactory::ClearFactory() {
  MatMulPrimitiveFwdFactory::GetInstance().Clear();
}

bool MatMulPrimitiveFwdFactory::DoNotCache() {
  return MatMulPrimitiveFwdFactory::GetInstance().do_not_cache_;
}

MatMulPrimitiveFwdFactory& MatMulPrimitiveFwdFactory::GetInstance() {
  static MatMulPrimitiveFwdFactory instance_;
  return instance_;
}
}  // namespace executor
