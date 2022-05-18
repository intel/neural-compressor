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

#include <exception>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "interface.hpp"

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
  const auto& wei_desc = ts_descs[0];
  const auto& src_desc = ts_descs[1];
  const auto& dst_desc = ts_descs[3];
  int dims = wei_desc.shape().size();
  int N = wei_desc.shape()[0];
  int K = wei_desc.shape()[1];
  int M = src_desc.shape()[1];
  const auto& dst_dt = dst_desc.dtype();
  auto attrs_map = op_desc.attrs();

  // runtime data alias
  const auto wei_data = static_cast<const bfloat16_t*>(rt_data[0]);
  const auto src_data = static_cast<const bfloat16_t*>(rt_data[1]);
  auto dst_data = const_cast<void*>(rt_data[3]);
  auto fp_dst_data = static_cast<float*>(dst_data);

  // Computing the kernel
  if (dims == 2) {
    for (int n = 0; n < N; ++n) {
#pragma omp parallel for
      for (int m = 0; m < M; ++m) {
#pragma omp parallel for
        for (int k = 0; k < K; ++k) {
          fp_dst_data[n * M + m] += make_fp32(wei_data[n * K + k]) * make_fp32(src_data[k * M + m]);
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
    auto buf1 = p.rt_data[3];
    auto size1 = p.op_desc.tensor_descs()[3].size();
    auto buf2 = q.rt_data[3];
    auto size2 = q.op_desc.tensor_descs()[3].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[3].dtype();
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  }
  return false;
}

class SpmmAMXX16KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmAMXX16KernelTest() {}
  virtual ~SpmmAMXX16KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmAMXX16KernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

template <typename T>
void prepare_sparse_data(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio) {
  uint32_t seed = 9527;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      weight[n * K + k] = make_bf16(rand_r(&seed) % 10 + 1);
    }
  }
  // sparsify a_mat
  for (int nb = 0; nb < N / n_blksize; ++nb) {
    for (int kb = 0; kb < K / k_blksize; ++kb) {
      bool fill_zero = rand_r(&seed) % 100 <= (dim_t)(ratio * 100);
      if (fill_zero) {
        for (int n = 0; n < n_blksize; ++n) {
          for (int k = 0; k < k_blksize; ++k) {
            weight[(nb * n_blksize + n) * K + kb * k_blksize + k] = make_bf16(0);
          }
        }
      }
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const dt& tensor_dt, dim_t rows, dim_t cols, bool is_clear = false,
                                                  bool is_sparse = false,
                                                  const std::vector<float>& ranges = {-10, 10}) {
  dim_t elem_num = rows * cols;
  dim_t bytes_size = elem_num * type_size[tensor_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    // prepare dst
    data_ptr = new float[elem_num];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (is_sparse) {
      // prepare sparse weight
      float ratio = 0.9;
      data_ptr = new bfloat16_t[elem_num];
      bfloat16_t* bf16_ptr = static_cast<bfloat16_t*>(data_ptr);
      prepare_sparse_data<bfloat16_t>(bf16_ptr, rows, cols, 16, 1, ratio);
    } else {
      // prepare dense activation
      data_ptr = new bfloat16_t[elem_num];
      bfloat16_t* bf16_ptr = static_cast<bfloat16_t*>(data_ptr);
      init_vector(bf16_ptr, elem_num, ranges[0], ranges[1]);
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(const jd::kernel_kind ker_kind, const jd::kernel_prop ker_prop,
                                         const jd::engine_kind eng_kind, const std::vector<tensor_desc>& ts_descs) {
  std::unordered_map<std::string, std::string> op_attrs;
  // Step 1: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  dim_t N = ts_descs[0].shape()[0];
  dim_t K = ts_descs[0].shape()[1];
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    bool is_clear = (index == 2 | index == 3) ? true : false;
    bool is_sparse = (index == 0) ? true : false;
    dim_t rows = ts_descs[index].shape()[0];
    dim_t cols = ts_descs[index].shape()[1];
    auto data_pair = make_data_obj(ts_descs[index].dtype(), rows, cols, is_clear, is_sparse);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // Step 2: sparse data encoding
  volatile bsr_data_t<bfloat16_t>* sparse_ptr = spns::reorder_to_bsr_amx<bfloat16_t, 32>(N, K, rt_data1[0]);
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  operator_desc an_op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // Config
  tensor_desc wei_desc;
  tensor_desc src_desc;
  tensor_desc bia_desc;
  tensor_desc dst_desc;

  // case: sparse: bf16xbf16=f32, weight(N, K) * activation(K, M)
  // = dst(N, M)
  wei_desc = {{64, 64}, dt::bf16, ft::bsr};
  src_desc = {{64, 64}, dt::bf16, ft::ab};
  bia_desc = {{64, 1}, dt::fp32, ft::ab};
  dst_desc = {{64, 64}, dt::fp32, ft::ab};
  cases.push_back({gen_case(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
                            {wei_desc, src_desc, bia_desc, dst_desc})});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SpmmAMXX16KernelTest, case_func());
}  // namespace jd
