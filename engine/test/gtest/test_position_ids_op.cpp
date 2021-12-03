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
#include "../../include/operators/position_ids.hpp"
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
  auto iter = attrs_map.find("mode");
  string mode;
  if (iter != attrs_map.end()) {
    mode = iter->second;
  }

  const vector<int64_t> src_shape = input[0]->shape();
  output[0]->set_shape(src_shape);

  Tensor* src = input[0];
  Tensor* dst = output[0];

  const vector<int64_t> dst_shape = dst->shape();
  const int batch_size = dst_shape[0];
  const int seq_len = dst_shape[1];

  int64_t src_size = src->size();
  const int32_t* src_data = static_cast<const int32_t*>(src->data());
  int32_t* dst_data = static_cast<int32_t*>(dst->mutable_data());

  if (mode == "roberta") {
    int32_t* equal_data = reinterpret_cast<int32_t*>(malloc(src_size * sizeof(int32_t)));
    memset(equal_data, 0, src_size * sizeof(int32_t));
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < seq_len; j++) {
        if (src_data[i * seq_len + j] != 1) {
          equal_data[i * seq_len + j] = 1;
        }
      }
    }
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < seq_len; j++) {
        if (j == 0) {
          dst_data[i * seq_len + j] = equal_data[i * seq_len + j];
        } else {
          dst_data[i * seq_len + j] = dst_data[i * seq_len + j - 1] + equal_data[i * seq_len + j];
        }
      }
    }
    for (int i = 0; i < src_size; i++) {
      dst_data[i] *= equal_data[i];
      dst_data[i] += 1;
    }
    free(equal_data);
  } else {
    LOG(ERROR) << "PositionIds mode is: " << mode << ", not supported. Only roberta is supported.";
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::PositionIdsOperator postion_ids(p.conf);
  postion_ids.Reshape(p.input, p.output);
  postion_ids.Forward(p.input, p.output);

  GetTrueData(q.input, q.output, q.conf);
  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class PositionIdsTest : public testing::TestWithParam<TestParams> {
 protected:
  PositionIdsTest() {}
  ~PositionIdsTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(PositionIdsTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape, std::string mode) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  std::vector<TensorConfig*> input_config = {src_config};
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"mode", mode}};

  AttrConfig* op_attr = new AttrConfig(attr_map);

  OperatorConfig op_config = OperatorConfig("position_ids", "fp32", input_config, output_config, op_attr);

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
    memcpy(reinterpret_cast<void*>(tensor_data_copy), tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;

  // case: roberta
  src_shape = {2, 128};
  cases.push_back({GenerateFp32Case({src_shape}, "roberta")});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, PositionIdsTest, CasesFp32());
