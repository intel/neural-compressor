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
#include "../../include/operators/softmax.hpp"
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
  auto src_tensor_shape = input[0]->shape();

  // dst shape
  int64_t dsize = src_tensor_shape.size();
  std::vector<int64_t> dst_shape(src_tensor_shape);
  output[0]->set_shape(dst_shape);
  float* dst_data = reinterpret_cast<float*>(output[0]->mutable_data());

  // attrs map
  auto attrs_map = conf.attributes();
  auto iter = attrs_map.find("axis");
  int axis = 0;
  if (iter != attrs_map.end() && iter->second != "") {
    axis = stoi(iter->second);
  } else {
    auto input0_tensor_shape = conf.input_tensors(0)->shape();
    axis = input0_tensor_shape.size() - 1;
  }
  std::vector<int64_t> reduce_shape(src_tensor_shape);
  reduce_shape[axis] = 1;
  float* maxsrc = new float[executor::Product(reduce_shape)];
  memset(maxsrc, 0, executor::Product(reduce_shape) * sizeof(float));
  float* sumexp = new float[executor::Product(reduce_shape)];
  memset(sumexp, 0, executor::Product(reduce_shape) * sizeof(float));

  // strides and data
  auto src_strides = executor::GetStrides(src_tensor_shape);
  auto dst_strides = src_strides;
  auto reduce_strides = executor::GetStrides(reduce_shape);
  const auto src_tensor_data = static_cast<const float*>(input[0]->data());
  if (dsize == 2) {
    // calculate max src
    for (int64_t i = 0; i < dst_shape[0]; ++i) {
      for (int64_t j = 0; j < dst_shape[1]; ++j) {
        int64_t src_idx = i * src_strides[0] + j * src_strides[1];
        int64_t reduce_idx =
            std::min(i, reduce_shape[0] - 1) * reduce_strides[0] + std::min(j, reduce_shape[1] - 1) * reduce_strides[1];
        maxsrc[reduce_idx] = std::max(src_tensor_data[src_idx], maxsrc[reduce_idx]);
      }
    }
    // calculate sum exp
    for (int64_t i = 0; i < dst_shape[0]; ++i) {
      for (int64_t j = 0; j < dst_shape[1]; ++j) {
        int64_t src_idx = i * src_strides[0] + j * src_strides[1];
        int64_t reduce_idx =
            std::min(i, reduce_shape[0] - 1) * reduce_strides[0] + std::min(j, reduce_shape[1] - 1) * reduce_strides[1];
        sumexp[reduce_idx] += exp(src_tensor_data[src_idx] - maxsrc[reduce_idx]);
      }
    }
    // calculate softmax
    for (int64_t i = 0; i < dst_shape[0]; ++i) {
      for (int64_t j = 0; j < dst_shape[1]; ++j) {
        int64_t src_idx = i * src_strides[0] + j * src_strides[1];
        int64_t reduce_idx =
            std::min(i, reduce_shape[0] - 1) * reduce_strides[0] + std::min(j, reduce_shape[1] - 1) * reduce_strides[1];
        dst_data[i * dst_strides[0] + j * dst_strides[1]] =
            exp(src_tensor_data[src_idx] - maxsrc[reduce_idx]) / sumexp[reduce_idx];
      }
    }
  }
  delete[] maxsrc;
  delete[] sumexp;
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::SoftmaxOperator smax_op(p.conf);
    smax_op.Reshape(p.input, p.output);
    smax_op.Forward(p.input, p.output);
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
                                        q.output[0]->size());
  }
  return false;
}

class SoftmaxOpTest : public testing::TestWithParam<TestParams> {
 protected:
  SoftmaxOpTest() {}
  ~SoftmaxOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SoftmaxOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  std::vector<TensorConfig*> input_config_vec = {src_config};
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("softmax", "fp32", input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const TensorConfig* a_tensor_config, int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size());

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(life_num);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(reinterpret_cast<void*>(tensor_data_copy), tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config, 2);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;

  // case: simple
  src_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src_shape}), false});
  // case: simple
  src_shape = {128, 64};
  cases.push_back({GenerateFp32Case({src_shape}), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SoftmaxOpTest, CasesFp32());
