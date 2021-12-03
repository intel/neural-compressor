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
#include "../../include/operators/strided_slice.hpp"
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

  executor::StridedSliceOperator strided_slice_op(p.conf);
  strided_slice_op.Reshape(p.input, p.output);
  strided_slice_op.Forward(p.input, p.output);

  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class StridedSliceOpTest : public testing::TestWithParam<TestParams> {
 protected:
  StridedSliceOpTest() {}
  ~StridedSliceOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(StridedSliceOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  std::vector<TensorConfig*> input_config_vec = {src_config};
  std::vector<int64_t> dst_shape = {src_shape[0], 1, src_shape[2]};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["begin_mask"] = "5";
  attr_map["ellipsis_mask"] = "0";
  attr_map["end_mask"] = "5";
  attr_map["new_axis_mask"] = "0";
  attr_map["shrink_axis_mask"] = "0";
  attr_map["begin"] = "0,0,0";
  attr_map["end"] = "0,1,0";
  attr_map["strides"] = "1,1,1";
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("strided_slice", "fp32", input_config_vec, output_config_vec, op_attr);

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

  auto src_tensors = make_tensor_obj(src_config, 2);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);
  auto dst_data_copy = static_cast<float*>(dst_tensor_copy->mutable_data());
  auto src_data_copy = static_cast<const float*>(src_tensors.second->data());
  for (int i = 0; i < dst_shape[0]; ++i) {
#pragma omp parallel for
    for (int j = 0; j < dst_shape[1]; ++j) {
#pragma omp simd
      for (int k = 0; k < dst_shape[2]; ++k) {
        dst_data_copy[i * dst_shape[1] * dst_shape[2] + j * dst_shape[2] + k] =
            src_data_copy[i * src_shape[1] * src_shape[2] + j * src_shape[2] + k];
      }
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
  src_shape = {64, 128, 768};  // dst = (64, 1, 768)
  cases.push_back({GenerateFp32Case({src_shape}), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, StridedSliceOpTest, CasesFp32());
