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

#include "../../include/common.hpp"
#include "../../include/conf.hpp"
#include "../../include/operators/inner_product.hpp"
#include "gtest/gtest.h"
using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;

struct OpArgs {
  std::vector<Tensor*> input;
  std::vector<Tensor*> output;
  OperatorConfig conf;
};

struct TestParams {
  std::pair<OpArgs, OpArgs> args;
  bool expect_to_fail;
};

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  auto attrs_map = conf.attributes();
  vector<int64_t> src0_perm;
  executor::StringSplit<int64_t>(&src0_perm, attrs_map["src0_perm"], ",");
  if (src0_perm.empty()) {  // default perm
    src0_perm = {0, 1};
  }
  vector<int64_t> src1_perm;
  executor::StringSplit<int64_t>(&src1_perm, attrs_map["src1_perm"], ",");
  if (src1_perm.empty()) {  // default perm
    src1_perm = {0, 1};
  }

  auto src_tensor_shape = input[0]->shape();
  const auto src_tensor_data = static_cast<const float*>(input[0]->data());
  auto wei_tensor_shape = input[1]->shape();
  const auto wei_tensor_data = static_cast<const float*>(input[1]->data());
  vector<int64_t> src_stride = executor::GetStrides(src_tensor_shape, src0_perm);
  vector<int64_t> wei_stride = executor::GetStrides(wei_tensor_shape, src1_perm);

  int M = src_tensor_shape[src0_perm[0]];
  int K = src_tensor_shape[src0_perm[1]];
  int N = wei_tensor_shape[src1_perm[0]];  // The weight_shape[0] of ip and matmul are opposite

  // dst shape
  output[0]->set_shape({M, N});
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    // #pragma omp simd
    for (int j = 0; j < N; ++j) {
      double value = 0;
#pragma omp simd
      for (int k = 0; k < K; ++k) {
        int src_idx = i * src_stride[0] + k * src_stride[1];
        int wei_idx = j * wei_stride[0] + k * wei_stride[1];
        value += static_cast<double>(src_tensor_data[src_idx]) * static_cast<double>(wei_tensor_data[wei_idx]);
      }
      dst_data[i * N + j] = value;
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::InnerProductOperator inner_product(p.conf);
    inner_product.Prepare(p.input, p.output);
    inner_product.Reshape(p.input, p.output);
    inner_product.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 5e-3);
  }
  return false;
}

class InnerProductTest : public testing::TestWithParam<TestParams> {
 protected:
  InnerProductTest() {}
  ~InnerProductTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(InnerProductTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape, std::string src1_perm,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  TensorConfig* src0_config = new TensorConfig("src", src0_shape);
  TensorConfig* src1_config = new TensorConfig("weight", src1_shape);
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> inputs_config = {src0_config, src1_config};
  if (append_op == "sum") {
    inputs_config.push_back(new TensorConfig("src2", input_shape[2]));
  }

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"src0_perm", ""}, {"src1_perm", src1_perm}, {"output_dtype", "fp32"}, {"append_op", append_op}};

  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("inner_product", "fp32", inputs_config, {dst_config}, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const TensorConfig* a_tensor_config) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(1);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size());

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(1);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };
  auto src0_tensors = make_tensor_obj(src0_config);
  auto src1_tensors = make_tensor_obj(src1_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  if (append_op == "sum") {
    auto src2_tensors = make_tensor_obj(inputs_config[2]);
    op_args.input.push_back(src2_tensors.first);
    op_args_copy.input.push_back(src2_tensors.second);
  }
  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> weight_shape;

  // case: simple
  src_shape = {1, 2};
  weight_shape = {1, 2};
  cases.push_back({GenerateFp32Case({src_shape, weight_shape}, "0,1"), false});
  // case: simple
  src_shape = {10, 20};
  weight_shape = {40, 20};
  cases.push_back({GenerateFp32Case({src_shape, weight_shape}, "0,1"), false});
  // case: simple
  src_shape = {10, 5};
  weight_shape = {5, 8};
  cases.push_back({GenerateFp32Case({src_shape, weight_shape}, "1,0"), false});
  // case: simple
  src_shape = {128, 768};
  weight_shape = {768, 512};
  cases.push_back({GenerateFp32Case({src_shape, weight_shape}, "1,0"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, InnerProductTest, CasesFp32());
