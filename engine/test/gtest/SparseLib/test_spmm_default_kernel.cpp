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

#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include "interface.hpp"
#include "gtest/gtest.h"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& src0_desc = ts_descs[ssd::WEI];
  const auto& src1_desc = ts_descs[ssd::SRC];
  const auto& bias_desc = ts_descs[ssd::BIAS];
  const auto& dst_desc = ts_descs[ssd::DST];
  int dims = src0_desc.shape().size();
  int M = src0_desc.shape()[0];
  int K = src0_desc.shape()[1];
  int N = src1_desc.shape()[1];
  const auto& left_dt = src0_desc.dtype();
  const auto& right_dt = src1_desc.dtype();
  const auto& dst_dt = dst_desc.dtype();
  bool has_bias = !bias_desc.shape().empty();
  auto attrs_map = op_desc.attrs();
  bool append_sum = (attrs_map["post_op"] == "append_sum");
  std::vector<int64_t> left_stride = {K, 1};
  std::vector<int64_t> right_stride = {N, 1};
  std::vector<int64_t> dst_stride = {N, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::WEI];
  const auto right_data = rt_data[ssd::SRC];
  const auto bias_data = static_cast<const int32_t*>(rt_data[ssd::BIAS]);
  auto dst_data = const_cast<void*>(rt_data[ssd::DST]);
  const auto scales_data = static_cast<const float*>(rt_data[ssd::SCALES]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto left_s8 = static_cast<const int8_t*>(left_data);

  auto right_fp32 = static_cast<const float*>(right_data);  // ptr alias
  auto right_u8 = static_cast<const uint8_t*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);

  auto dst_fp32 = static_cast<float*>(dst_data);  // ptr alias
  auto dst_s32 = static_cast<int32_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);

  // Computing the kernel
  if (dims == 2) {
    for (int i = 0; i < M; ++i) {
      #pragma omp parallel for
      for (int j = 0; j < N; ++j) {
        float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
        #pragma omp simd
        for (int k = 0; k < K; ++k) {
          int idx0 = i * left_stride[0] + k * left_stride[1];
          int idx1 = k * right_stride[0] + j * right_stride[1];
          auto left_k = (left_dt == dt::fp32) ? left_fp32[idx0] : ((left_dt == dt::u8) ?
            left_u8[idx0] : ((left_dt == dt::s8) ? left_s8[idx0] : 0));
          auto right_k = (right_dt == dt::fp32) ? right_fp32[idx1] : ((right_dt == dt::u8) ?
            right_u8[idx1] : ((right_dt == dt::s8) ? right_s8[idx1] : 0));
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        int dst_idx = i * dst_stride[0] + j * dst_stride[1];
        int scale_idx = i;
        if (append_sum) {
          value = value * scales_data[scale_idx] + dst_fp32[dst_idx];
        }

        // Quantize dst data
        if (dst_dt == dt::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == dt::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == dt::s8) {
          int32_t data = nearbyint(value * scales_data[scale_idx]);
          data = data < -128 ? -128 : data;
          data = data > 127 ? 127 : data;
          dst_s8[dst_idx] = static_cast<int8_t>(data);
        }
      }
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    const auto& op_desc = p.op_desc;
    sparse_matmul_desc spmm_desc(op_desc);
    sparse_matmul spmm_kern(spmm_desc);
    spmm_kern.execute(p.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.rt_data);
    auto buf1 = p.rt_data[ssd::DST];
    auto size1 = p.op_desc.tensor_descs()[ssd::DST].size();
    auto buf2 = q.rt_data[ssd::DST];
    auto size2 = q.op_desc.tensor_descs()[ssd::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST].dtype();
    if (dst_type == dt::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
    }
  }
  return false;
}

class SpmmDefaultKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmDefaultKernelTest() {}
  virtual ~SpmmDefaultKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmDefaultKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

template <typename T>
void prepare_sparse_data(T* vector_data, std::vector<int64_t> a_shape) {
  int64_t M = a_shape[0];
  int64_t K = a_shape[1];
  // Blocks zeros in the M dimension.
  int64_t BLOCK = 4;
  int64_t nums = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int64_t block_nums = nums / BLOCK;
  float sparse_ratio = 0.7;
  std::unordered_set<int64_t> zero_block_index;
  uint32_t seed = 123;
  while (zero_block_index.size() < block_nums * sparse_ratio) {
    zero_block_index.insert((rand_r(&seed) % (block_nums - 1)));
  }
  for (const auto& i : zero_block_index) {
    for (int j = 0; j < BLOCK; ++j) {
      int64_t zero_idx = i * BLOCK + j;
      int64_t zero_row = zero_idx / M;
      int64_t zero_col = zero_idx % M;
      // vector_data is (M, K). Block zeros is continuous in M-dim.
      vector_data[zero_col * K + zero_row] = 0;
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape,
    const dt& a_dt, bool is_clear = false, bool is_sparse = false,
    const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (is_sparse) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        for (int i = 0; i < elem_num; ++i) {  // firstly remove zero.
          s8_ptr[i] = (s8_ptr[i] == 0) ? 1 : s8_ptr[i];
        }
        prepare_sparse_data(s8_ptr, a_shape);
      }
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(const jd::kernel_kind ker_kind, const jd::kernel_prop ker_prop,
  const jd::engine_kind eng_kind, const std::vector<tensor_desc>& ts_descs,
  const std::string& mkn_blocks = "1,1,1", const std::string& tile_shape = "4,4", std::string post_op = "") {
  // Step 1: Construct operator config
  std::unordered_map<std::string, std::string> op_attrs = {
    {"mkn_blocks", mkn_blocks},
    {"tile_shape", tile_shape},
    {"post_op", post_op}
  };
  bool append_sum = (op_attrs["post_op"] == "append_sum");

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    bool is_clear = (index == ssd::DST && !append_sum) ? true : false;
    bool is_sparse = (index == ssd::WEI) ? true : false;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair =  make_data_obj(ts_descs[index].shape(), ts_descs[index].dtype(),
                                    is_clear, is_sparse, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // Step 3: sparse data encoding
  int M = ts_descs[ssd::WEI].shape()[0];
  int K = ts_descs[ssd::WEI].shape()[1];
  auto sparse_ptr = spns::reorder_to<int8_t>(M, K, rt_data1[ssd::WEI], format_type::csrp);
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  operator_desc an_op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);

  // Step 4: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // Config
  tensor_desc src0_desc;
  tensor_desc src1_desc;
  tensor_desc bias_desc;
  tensor_desc dst_desc;
  tensor_desc scales_desc;
  std::string mkn_blocks;
  std::string tile_shape;

  /* bert-mini config. case: spmm: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N) */
  // when M = 256, K = 256, N = 128
  src0_desc = {{256, 256}, dt::s8, ft::csrp};
  src1_desc = {{256, 128}, dt::u8, ft::ab};
  bias_desc = {{256, 1}, dt::s32, ft::ab};
  dst_desc = {{256, 128}, dt::s8, ft::ab};
  scales_desc = {{256, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 256, K = 256, N = 384
  src0_desc = {{256, 256}, dt::s8, ft::csrp};
  src1_desc = {{256, 384}, dt::u8, ft::ab};
  bias_desc = {{256, 1}, dt::s32, ft::ab};
  dst_desc = {{256, 384}, dt::s8, ft::ab};
  scales_desc = {{256, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 256, K = 1024, N = 128
  src0_desc = {{256, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 128}, dt::u8, ft::ab};
  bias_desc = {{256, 1}, dt::s32, ft::ab};
  dst_desc = {{256, 128}, dt::s8, ft::ab};
  scales_desc = {{256, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 256, K = 1024, N = 384
  src0_desc = {{256, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 384}, dt::u8, ft::ab};
  bias_desc = {{256, 1}, dt::s32, ft::ab};
  dst_desc = {{256, 384}, dt::s8, ft::ab};
  scales_desc = {{256, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 1024, K = 256, N = 128
  src0_desc = {{1024, 256}, dt::s8, ft::csrp};
  src1_desc = {{256, 128}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 128}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 1024, K = 256, N = 384
  src0_desc = {{1024, 256}, dt::s8, ft::csrp};
  src1_desc = {{256, 384}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 384}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  /* bert-base config. case: spmm: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N) */
  // when M = 768, K = 768, N = 128
  src0_desc = {{768, 768}, dt::s8, ft::csrp};
  src1_desc = {{768, 128}, dt::u8, ft::ab};
  bias_desc = {{768, 1}, dt::s32, ft::ab};
  dst_desc = {{768, 128}, dt::s8, ft::ab};
  scales_desc = {{768, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 768, K = 768, N = 384
  src0_desc = {{768, 768}, dt::s8, ft::csrp};
  src1_desc = {{768, 384}, dt::u8, ft::ab};
  bias_desc = {{768, 1}, dt::s32, ft::ab};
  dst_desc = {{768, 384}, dt::s8, ft::ab};
  scales_desc = {{768, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 768, K = 3072, N = 128
  src0_desc = {{768, 3072}, dt::s8, ft::csrp};
  src1_desc = {{3072, 128}, dt::u8, ft::ab};
  bias_desc = {{768, 1}, dt::s32, ft::ab};
  dst_desc = {{768, 128}, dt::s8, ft::ab};
  scales_desc = {{768, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 768, K = 3072, N = 384
  src0_desc = {{768, 3072}, dt::s8, ft::csrp};
  src1_desc = {{3072, 384}, dt::u8, ft::ab};
  bias_desc = {{768, 1}, dt::s32, ft::ab};
  dst_desc = {{768, 384}, dt::s8, ft::ab};
  scales_desc = {{768, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 3072, K = 768, N = 128
  src0_desc = {{3072, 768}, dt::s8, ft::csrp};
  src1_desc = {{768, 128}, dt::u8, ft::ab};
  bias_desc = {{3072, 1}, dt::s32, ft::ab};
  dst_desc = {{3072, 128}, dt::s8, ft::ab};
  scales_desc = {{3072, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 3072, K = 768, N = 384
  src0_desc = {{3072, 768}, dt::s8, ft::csrp};
  src1_desc = {{768, 384}, dt::u8, ft::ab};
  bias_desc = {{3072, 1}, dt::s32, ft::ab};
  dst_desc = {{3072, 384}, dt::s8, ft::ab};
  scales_desc = {{3072, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  /* bert-large config. case: spmm: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N) */
  // when M = 1024, K = 1024, N = 128
  src0_desc = {{1024, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 128}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 128}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 1024, K = 1024, N = 384
  src0_desc = {{1024, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 384}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 384}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 1024, K = 4096, N = 128
  src0_desc = {{1024, 4096}, dt::s8, ft::csrp};
  src1_desc = {{4096, 128}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 128}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 1024, K = 4096, N = 384
  src0_desc = {{1024, 4096}, dt::s8, ft::csrp};
  src1_desc = {{4096, 384}, dt::u8, ft::ab};
  bias_desc = {{1024, 1}, dt::s32, ft::ab};
  dst_desc = {{1024, 384}, dt::s8, ft::ab};
  scales_desc = {{1024, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 4096, K = 1024, N = 128
  src0_desc = {{4096, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 128}, dt::u8, ft::ab};
  bias_desc = {{4096, 1}, dt::s32, ft::ab};
  dst_desc = {{4096, 128}, dt::s8, ft::ab};
  scales_desc = {{4096, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // when M = 4096, K = 1024, N = 384
  src0_desc = {{4096, 1024}, dt::s8, ft::csrp};
  src1_desc = {{1024, 384}, dt::u8, ft::ab};
  bias_desc = {{4096, 1}, dt::s32, ft::ab};
  dst_desc = {{4096, 384}, dt::s8, ft::ab};
  scales_desc = {{4096, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // case: sparse: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
  src0_desc = {{32, 32}, dt::s8, ft::csrp};
  src1_desc = {{32, 128}, dt::u8, ft::ab};
  bias_desc = {{32, 1}, dt::s32, ft::ab};
  dst_desc = {{32, 128}, dt::s8, ft::ab};
  scales_desc = {{32, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // case: sparse: s8xu8+s32=s8, n_blocks != 1, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
  src0_desc = {{32, 32}, dt::s8, ft::csrp};
  src1_desc = {{32, 128}, dt::u8, ft::ab};
  bias_desc = {{32, 1}, dt::s32, ft::ab};
  dst_desc = {{32, 128}, dt::s8, ft::ab};
  scales_desc = {{32, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,2";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // case: sparse: s8xu8+s32=s8, k_blocks != 1, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
  src0_desc = {{32, 32}, dt::s8, ft::csrp};
  src1_desc = {{32, 128}, dt::u8, ft::ab};
  bias_desc = {{32, 1}, dt::s32, ft::ab};
  dst_desc = {{32, 128}, dt::s8, ft::ab};
  scales_desc = {{32, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,2,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape), false});

  // case: sparse: s8xu8+s32+append_fp32=fp32,
  // weight(M, K) * activation(K, N) + bias(M, 1) + append_sum(M, N) = dst(M, N)
  src0_desc = {{32, 32}, dt::s8, ft::csrp};
  src1_desc = {{32, 128}, dt::u8, ft::ab};
  bias_desc = {{32, 1}, dt::s32, ft::ab};
  dst_desc = {{32, 128}, dt::fp32, ft::ab};
  scales_desc = {{32, 1}, dt::fp32, ft::ab};
  mkn_blocks = "1,1,1";
  tile_shape = "4,4";
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
    {src0_desc, src1_desc, bias_desc, dst_desc, scales_desc}, mkn_blocks, tile_shape, "append_sum"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SpmmDefaultKernelTest, case_func());
}  // namespace jd
