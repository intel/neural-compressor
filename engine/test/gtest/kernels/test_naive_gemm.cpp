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

#include <map>
#include <string>

#include "../../../executor/include/kernels/jit_naive_matmul.hpp"
#include "../../../include/common.hpp"
#include "gtest/gtest.h"

struct OpArgs {
  int M;
  int K;
  int N;
  float* src;
  float* wgt;
  float* dst;
};

struct TestParams {
  std::pair<OpArgs, OpArgs> args;
};

void GetTrueData(const int M, const int K, const int N, const float* input, const float* weight, float* output) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float value = 0;
      for (int k = 0; k < K; ++k) {
        int src_idx = i * K + k;
        int wgt_idx = k * N + j;
        value += input[src_idx] * weight[wgt_idx];
      }
      int dst_idx = i * N + j;
      output[dst_idx] = value;
    }
  }
  return;
}

bool CheckResult(const TestParams& t) {
  auto& p = t.args.first;
  auto& q = t.args.second;
  const int M = p.M;
  const int N = p.N;
  const int K = p.K;
  try {
    executor::naive_gemm_params_t param;
    param.src = reinterpret_cast<void*>(p.src);
    param.weight = reinterpret_cast<void*>(p.wgt);
    param.dst = reinterpret_cast<void*>(p.dst);
    executor::jit_naive_matmul op(M, N, K);
    op(param);
  } catch (...) {
    return false;
  }
  GetTrueData(M, K, N, q.src, q.wgt, q.dst);
  // Should compare buffer with different addresses
  EXPECT_NE(p.dst, q.dst);
  return executor::CompareData<float>(reinterpret_cast<float*>(p.dst), M * N, reinterpret_cast<float*>(q.dst), M * N,
                                      1e-3);
}

class NaiveGEMMTest : public testing::TestWithParam<TestParams> {
 protected:
  NaiveGEMMTest() {}
  ~NaiveGEMMTest() override {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(NaiveGEMMTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
  free(t.args.first.src);
  free(t.args.first.wgt);
  free(t.args.first.dst);
  free(t.args.second.src);
  free(t.args.second.wgt);
  free(t.args.second.dst);
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const int M, const int K, const int N) {
  float* src = reinterpret_cast<float*>(malloc(sizeof(float) * M * K));
  float* wgt = reinterpret_cast<float*>(malloc(sizeof(float) * K * N));
  float* dst = reinterpret_cast<float*>(malloc(sizeof(float) * M * N));
  float* src_ref = reinterpret_cast<float*>(malloc(sizeof(float) * M * K));
  float* wgt_ref = reinterpret_cast<float*>(malloc(sizeof(float) * K * N));
  float* dst_ref = reinterpret_cast<float*>(malloc(sizeof(float) * M * N));

  const unsigned int seed_src = 12345;
  const unsigned int seed_wgt = 54321;

  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      unsigned int seed_src_temp = seed_src + m * K + k;
      src[m * K + k] = rand_r(&seed_src_temp) % 10;
      src_ref[m * K + k] = src[m * K + k];
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      unsigned int seed_wgt_temp = seed_wgt + k * N + n;
      wgt[k * N + n] = rand_r(&seed_wgt_temp) % 10;
      wgt_ref[k * N + n] = wgt[k * N + n];
    }
  }
  memset(dst, 0, sizeof(float) * M * N);
  memset(dst_ref, 0, sizeof(float) * M * N);

  OpArgs p = {M, K, N, src, wgt, dst};
  OpArgs q = {M, K, N, src_ref, wgt_ref, dst_ref};
  return {p, q};
}

static auto CasesFp32 = []() {
  std::vector<TestParams> cases;

  // case: simple
  cases.push_back({GenerateFp32Case(4, 4, 48)});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, NaiveGEMMTest, CasesFp32());
