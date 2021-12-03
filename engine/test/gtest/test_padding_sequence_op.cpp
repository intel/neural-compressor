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
#include "../../include/operators/padding_sequence.hpp"
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

  // attrs map
  auto attrs_map = conf.attributes();
  std::vector<int64_t> dst_shape_vec;
  executor::StringSplit<int64_t>(&dst_shape_vec, attrs_map["dst_shape"], ",");
  std::vector<int64_t> dims_vec;
  executor::StringSplit<int64_t>(&dims_vec, attrs_map["dims"], ",");

  // dst shape
  int padding_idx = 0;
  int broadcast_idx = 0;
  vector<int64_t> dst_shape = dst_shape_vec;
  for (int i = 0; i < dst_shape_vec.size(); ++i) {
    if (dst_shape_vec[i] == -1 && padding_idx < src_tensor_shape.size()) {
      dst_shape[i] = src_tensor_shape[padding_idx++];
      continue;
    }
    if (dst_shape_vec[i] == 0 && broadcast_idx < dims_vec.size()) {
      dst_shape[i] = src_tensor_shape[dims_vec[broadcast_idx++]];
      continue;
    }
  }
  output[0]->set_shape(dst_shape);
  float* dst_data = static_cast<float*>(output[0]->mutable_data());

  int64_t bs = dst_shape[0];
  int64_t seq = dst_shape.back();
  // pad_dst_shape = {bs, broadcast_num, ..., broadcast_num, seq} or {bs, seq}
  vector<int64_t> pad_dst_shape = {bs, seq};
  // pad to 3 dimensions because that is what the runtime code requires
  int64_t runtime_dims = 3;
  int64_t broadcast_nums = 1;
  if (dst_shape.size() >= runtime_dims) {
    broadcast_nums =
        std::accumulate(dst_shape.begin() + 1, dst_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
  }
  pad_dst_shape.insert(pad_dst_shape.begin() + 1, broadcast_nums);

  int64_t batch_size = pad_dst_shape[0];
  int64_t row = pad_dst_shape[1];
  int64_t col = pad_dst_shape[2];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < row; ++j) {
      for (int k = 0; k < col; ++k) {
        dst_data[i * (row * col) + j * col + k] = (k < col / 2) ? 0 : -10000;
      }
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::PaddingSequenceOperator padding_seq_op(p.conf);
  padding_seq_op.Reshape(p.input, p.output);
  padding_seq_op.Forward(p.input, p.output);

  GetTrueData(q.input, q.output, q.conf);
  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class PaddingSequenceOpTest : public testing::TestWithParam<TestParams> {
 protected:
  PaddingSequenceOpTest() {}
  ~PaddingSequenceOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(PaddingSequenceOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string attr_dst_shape = "-1,-1", std::string attr_dims = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("input_mask", src_shape);
  std::vector<TensorConfig*> input_config_vec = {src_config};
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["dst_shape"] = attr_dst_shape;
  attr_map["dims"] = attr_dims;
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("padding_sequence", "fp32", input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const TensorConfig* a_tensor_config, int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = static_cast<float*>(a_tensor->mutable_data());
    int batch_size = a_tensor->shape()[0];
    int seq_len = a_tensor->shape()[1];
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        tensor_data[i * seq_len + j] = (j < seq_len / 2) ? 1 : 0;
      }
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

  OpArgs op_args = {{src_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;

  // case: simple, pad to 2D
  src_shape = {16, 32};
  cases.push_back({GenerateFp32Case({src_shape}, "-1,-1"), false});
  // case: simple, pad to 2D
  src_shape = {128, 48};
  cases.push_back({GenerateFp32Case({src_shape}, "-1,-1"), false});
  // case: simple, pad to 3D
  src_shape = {32, 128};
  cases.push_back({GenerateFp32Case({src_shape}, "-1,1,-1"), false});
  // case: simple, pad to 4D, and broadcast
  src_shape = {32, 128};
  cases.push_back({GenerateFp32Case({src_shape}, "-1,2,0,-1", "1"), false});
  // case: simple, pad to 5D
  src_shape = {32, 128};
  cases.push_back({GenerateFp32Case({src_shape}, "-1,1,1,1,-1"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, PaddingSequenceOpTest, CasesFp32());
