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
#include "../../include/operators/one_hot.hpp"
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

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::OnehotOperator onehot_op(p.conf);
  onehot_op.Reshape(p.input, p.output);
  onehot_op.Forward(p.input, p.output);

  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class OnehotOpTest : public testing::TestWithParam<TestParams> {
 protected:
  OnehotOpTest() {}
  ~OnehotOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(OnehotOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("indices_tensor", src_shape, "int32");
  std::vector<TensorConfig*> input_config_vec = {src_config};
  int64_t depth = 2;
  int64_t on_value = 1;
  int64_t off_value = 0;
  std::vector<int64_t> dst_shape = {src_shape[0], depth};  // axis = -1, depth = 2
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["axis"] = "-1";
  attr_map["depth"] = std::to_string(depth);
  attr_map["on_value"] = std::to_string(on_value);
  attr_map["off_value"] = std::to_string(off_value);
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("one_hot", "fp32", input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const TensorConfig* a_tensor_config, int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = static_cast<int32_t*>(a_tensor->mutable_data());
    uint32_t seed = 123;
    for (int i = 0; i < a_tensor->size(); ++i) {
      tensor_data[i] = (int32_t)(rand_r(&seed) % 3);
    }

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(life_num);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);
  auto dst_data_copy = static_cast<float*>(dst_tensor_copy->mutable_data());
  auto src_data_copy = (const int32_t*)src_tensors.second->data();
  for (int i = 0; i < dst_shape[0]; ++i) {
    for (int j = 0; j < depth; ++j) {
      dst_data_copy[i * depth + j] = (j == src_data_copy[i]) ? on_value : off_value;
    }
  }

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
  src_shape = {64};
  cases.push_back({GenerateFp32Case({src_shape}), false});
  // case: simple
  src_shape = {8192};
  cases.push_back({GenerateFp32Case({src_shape}), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, OnehotOpTest, CasesFp32());
