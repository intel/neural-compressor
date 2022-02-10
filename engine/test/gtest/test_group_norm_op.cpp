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
#include "../../include/operators/group_norm.hpp"
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

void GroupNormRefGT(const float* src_data, const float* gamma_data, const float* beta_data, float* dst_data,
                    const vector<int64_t>& src_shape, const float eps, const int64_t group, const int64_t channels,
                    const bool affine) {
  // x = (x - mean) / sqrt(var + eps) * gamma + beta
  const int64_t batch_size = src_shape[0];
  int64_t map_size = 1;
  for (int i = 2; i < src_shape.size(); ++i) {
    map_size *= src_shape[i];
  }
  const int64_t channels_per_group = channels / group;

#pragma omp parallel for
  for (int64_t n = 0; n < batch_size; n++) {
    const float* src_single_data = src_data + n * channels * map_size;
    float* dst_single_data = dst_data + n * channels * map_size;
#pragma omp simd
    for (int64_t g = 0; g < group; g++) {
      const float* src_group_data = src_single_data + g * channels_per_group * map_size;
      float* dst_group_data = dst_single_data + g * channels_per_group * map_size;
      // mean and var
      float sum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          sum += ptr[i];
        }
      }
      float mean = sum / (channels_per_group * map_size);

      float sqsum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          float tmp = ptr[i] - mean;
          sqsum += tmp * tmp;
        }
      }
      float var = sqsum / (channels_per_group * map_size);

      for (int64_t q = 0; q < channels_per_group; q++) {
        float a;
        float b;
        if (affine) {
          float gamma = gamma_data[g * channels_per_group + q];
          float beta = beta_data[g * channels_per_group + q];

          a = static_cast<float>(gamma / sqrt(var + eps));
          b = -mean * a + beta;
        } else {
          a = static_cast<float>(1.f / (sqrt(var + eps)));
          b = -mean * a;
        }

        const float* ptr = src_group_data + q * map_size;
        float* dst_ptr = dst_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          dst_ptr[i] = ptr[i] * a + b;
        }
      }
    }
  }
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  // config parse
  float epsilon = 1e-05;
  int64_t group = 1;
  int64_t channels;
  bool affine = false;
  auto attrs_map = conf.attributes();
  auto iter = attrs_map.find("epsilon");
  if (iter != attrs_map.end()) {
    epsilon = executor::StringToNum<float>(attrs_map["epsilon"]);
  }
  iter = attrs_map.find("group");
  if (iter != attrs_map.end()) {
    group = executor::StringToNum<int64_t>(attrs_map["group"]);
  }
  iter = attrs_map.find("channels");
  if (iter != attrs_map.end()) {
    channels = executor::StringToNum<int64_t>(attrs_map["channels"]);
  }
  Tensor* src = input[0];
  Tensor* gamma = input[1];
  Tensor* beta = input[2];
  Tensor* dst = output[0];
  const vector<int64_t> src_shape = src->shape();
  dst->set_dtype(src->dtype());
  dst->set_shape(src_shape);
  assert(src->dtype() == "fp32");
  assert(src_shape.size() > 2);
  assert(gamma->shape()[0] == channels);
  assert(beta->shape()[0] == channels);
  const float* src_data = static_cast<const float*>(src->data());
  const float* gamma_data = static_cast<const float*>(gamma->data());
  const float* beta_data = static_cast<const float*>(beta->data());
  for (int64_t i = 0; i < channels; ++i) {
    if (gamma_data[i] != 1.f || beta_data[i] != 0.f) {
      affine = true;
      break;
    }
  }
  float* dst_data = static_cast<float*>(dst->mutable_data());
  GroupNormRefGT(src_data, gamma_data, beta_data, dst_data, src_shape, epsilon, group, channels, affine);
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  executor::GroupNormOperator group_norm(p.conf);
  group_norm.Prepare(p.input, p.output);
  group_norm.Reshape(p.input, p.output);
  group_norm.Forward(p.input, p.output);

  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size());
  }
  return false;
}

class GroupNormTest : public testing::TestWithParam<TestParams> {
 protected:
  GroupNormTest() {}
  ~GroupNormTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GroupNormTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           const std::string& epsilon, const std::string& group,
                                           const std::string& channels) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  const auto& gamma_shape = input_shape[1];
  const auto& beta_shape = input_shape[2];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  TensorConfig* gamma_config = new TensorConfig("gamma", gamma_shape);
  TensorConfig* beta_config = new TensorConfig("beta", beta_shape);
  std::vector<TensorConfig*> input_config_vec = {src_config, gamma_config, beta_config};
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"epsilon", epsilon}, {"group", group}, {"channels", channels}};
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("group_norm", "fp32", input_config_vec, output_config_vec, op_attr);

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
  auto gamma_tensors = make_tensor_obj(gamma_config);
  auto beta_tensors = make_tensor_obj(beta_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, gamma_tensors.first, beta_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second, gamma_tensors.second, beta_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;
  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> gamma_shape;
  std::vector<int64_t> beta_shape;
  std::string epsilon;
  std::string group;
  std::string channels;

  // case: 3d group norm
  src_shape = {1, 4, 2};
  gamma_shape = {4};
  beta_shape = {4};
  epsilon = "0.00001";
  group = "1";
  channels = "4";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}, epsilon, group, channels), false});

  // case: 3d group norm, batch != 1
  src_shape = {3, 4, 2};
  gamma_shape = {4};
  beta_shape = {4};
  epsilon = "0.00001";
  group = "1";
  channels = "4";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}, epsilon, group, channels), false});

  // case: 3d group norm, batch != 1, group != 1
  src_shape = {3, 4, 2};
  gamma_shape = {4};
  beta_shape = {4};
  epsilon = "0.00001";
  group = "2";
  channels = "4";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}, epsilon, group, channels), false});

  // case: 4d group norm, batch != 1, group != 1
  src_shape = {3, 4, 2, 3};
  gamma_shape = {4};
  beta_shape = {4};
  epsilon = "0.0001";
  group = "2";
  channels = "4";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}, epsilon, group, channels), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, GroupNormTest, CasesFp32());
