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

#include <math.h>

#include <map>
#include <string>

#include "../../include/common.hpp"
#include "../../include/conf.hpp"
#include "../../include/operators/reorder.hpp"
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
  int64_t dsize = src_tensor_shape.size();
  auto src_strides = executor::GetStrides(src_tensor_shape);
  const auto src_tensor_data = static_cast<const float*>(input[0]->data());

  // dst shape
  auto attrs_map = conf.attributes();
  vector<int64_t> dst_perm;
  executor::StringSplit<int64_t>(&dst_perm, attrs_map["dst_perm"], ",");
  std::vector<int64_t> dst_shape = src_tensor_shape;
  std::vector<int64_t> dst_stride_before_postTrans = executor::GetStrides(dst_shape, dst_perm);

  output[0]->set_shape(dst_shape);
  float* dst_data = static_cast<float*>(output[0]->mutable_data());

  // attrs map
  int h = src_tensor_shape[0];
  int w = src_tensor_shape[1];

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      dst_data[i * h + j] = src_tensor_data[j * w + i];
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    int dsize = p.input[0]->size();
    executor::ReorderOperator reorder_op(p.conf);
    reorder_op.Prepare(p.input, p.output);
    reorder_op.Reshape(p.input, p.output);
    reorder_op.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size());
  }
  return false;
}

class ReorderOpTest : public testing::TestWithParam<TestParams> {
 protected:
  ReorderOpTest() {}
  ~ReorderOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(ReorderOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> inputs_config = {src_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["src_perm"] = "0,1";
  attr_map["dst_perm"] = "1,0";
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("reorder", "fp32", inputs_config, {dst_config}, op_attr);

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
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> gamma_shape;
  std::vector<int64_t> beta_shape;

  // case: simple 2D
  src_shape = {2, 2};
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}), false});
  // // case: expect fail
  // src_shape = {4, 4, 4, 8, 16, 32};
  // gamma_shape = {1, 32};
  // beta_shape = {1, 32};
  // cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}), true});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, ReorderOpTest, CasesFp32());
