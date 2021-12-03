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
#include "../../include/operators/binary_add.hpp"
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
  auto src0_tensor_shape = input[0]->shape();
  auto src1_tensor_shape = input[1]->shape();

  // dst shape
  int64_t dsize = src0_tensor_shape.size();
  std::vector<int64_t> dst_shape(dsize);
  for (int64_t i = 0; i < dsize; ++i) {
    dst_shape[i] = std::max(src0_tensor_shape[i], src1_tensor_shape[i]);
  }
  output[0]->set_shape(dst_shape);

  // attrs map
  auto attrs_map = conf.attributes();
  auto iter = attrs_map.find("append_op");
  bool append_sum = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  if (append_sum) {
    output[0]->set_data(const_cast<void*>(input[2]->data()));
  } else {
    void* ptr = calloc(output[0]->size(), sizeof(float));
    output[0]->unref_data(true);
    output[0]->set_data(ptr);
    // memset(output[0]->mutable_data(), 0, output[0]->size() * sizeof(float));
  }
  float* dst_data = static_cast<float*>(output[0]->mutable_data());

  // strides and data
  auto src0_strides = executor::GetStrides(src0_tensor_shape);
  auto src1_strides = executor::GetStrides(src1_tensor_shape);
  auto dst_strides = executor::GetStrides(dst_shape);
  const auto src0_tensor_data = static_cast<const float*>(input[0]->data());
  const auto src1_tensor_data = static_cast<const float*>(input[1]->data());
  if (dsize == 2) {
    for (int64_t i = 0; i < dst_shape[0]; ++i) {
      for (int64_t j = 0; j < dst_shape[1]; ++j) {
        int64_t src0_idx = std::min(i, src0_tensor_shape[0] - 1) * src0_strides[0] +
                           std::min(j, src0_tensor_shape[1] - 1) * src0_strides[1];
        int64_t src1_idx = std::min(i, src1_tensor_shape[0] - 1) * src1_strides[0] +
                           std::min(j, src1_tensor_shape[1] - 1) * src1_strides[1];
        dst_data[i * dst_strides[0] + j * dst_strides[1]] += src0_tensor_data[src0_idx] + src1_tensor_data[src1_idx];
      }
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::BinaryAddOperator badd_op(p.conf);
    badd_op.Reshape(p.input, p.output);
    badd_op.Forward(p.input, p.output);
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

class BinaryAddOpTest : public testing::TestWithParam<TestParams> {
 protected:
  BinaryAddOpTest() {}
  ~BinaryAddOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(BinaryAddOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  TensorConfig* src0_config = new TensorConfig("src0", src0_shape);
  TensorConfig* src1_config = new TensorConfig("src1", src1_shape);
  std::vector<TensorConfig*> input_config_vec = {src0_config, src1_config};
  if (append_op == "sum") {
    input_config_vec.push_back(new TensorConfig("src2", input_shape[2]));
  }
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["append_op"] = append_op;
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("binary_add", "fp32", input_config_vec, output_config_vec, op_attr);

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
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src0_tensors = make_tensor_obj(src0_config, 2);
  auto src1_tensors = make_tensor_obj(src1_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  if (append_op == "sum") {
    auto src2_tensors = make_tensor_obj(input_config_vec[2], 2);
    op_args.input.push_back(src2_tensors.first);
    op_args_copy.input.push_back(src2_tensors.second);
  }
  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;

  // case: simple
  src0_shape = {16, 32};
  src1_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), false});
  // case: broadcast
  src0_shape = {16, 1};
  src1_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), false});
  // case: expect fail
  src0_shape = {16, 2, 2};
  src1_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), true});
  // case: append_op:sum
  src0_shape = {16, 32};
  src1_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, {16, 32}}, "sum"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, BinaryAddOpTest, CasesFp32());
