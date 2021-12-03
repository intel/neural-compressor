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
#include "../../include/operators/layer_norm.hpp"
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
  auto gamma_tensor_shape = input[1]->shape();
  auto beta_tensor_shape = input[2]->shape();

  // dst shape
  int64_t dsize = src_tensor_shape.size();
  std::vector<int64_t> dst_shape(src_tensor_shape);
  output[0]->set_shape(dst_shape);
  float* dst_data = static_cast<float*>(output[0]->mutable_data());

  // attrs map
  auto attrs_map = conf.attributes();
  float epsilon = executor::StringToNum<float>(attrs_map["epsilon"]);
  auto mu_shape = vector<int64_t>(src_tensor_shape.begin(), src_tensor_shape.end() - 1);
  auto sigma2_shape = mu_shape;
  float* mu = new float[executor::Product(mu_shape)];
  float* sigma2 = new float[executor::Product(sigma2_shape)];

  // strides and data
  auto src_strides = executor::GetStrides(src_tensor_shape);
  auto dst_strides = src_strides;
  const auto src_tensor_data = static_cast<const float*>(input[0]->data());
  const auto gamma_tensor_data = static_cast<const float*>(input[1]->data());
  const auto beta_tensor_data = static_cast<const float*>(input[2]->data());
  if (dsize == 2) {
    // calculate mu
    for (int64_t i = 0; i < src_tensor_shape[0]; ++i) {
      mu[i] = 0;
      auto p = src_tensor_data + i * src_strides[0];
      for (int64_t j = 0; j < src_tensor_shape[1]; ++j) {
        mu[i] += p[j];
      }
      mu[i] = mu[i] / src_tensor_shape[1];
    }
    // calculate sigma2
    for (int64_t i = 0; i < src_tensor_shape[0]; ++i) {
      sigma2[i] = 0;
      auto p = src_tensor_data + i * src_strides[0];
      for (int64_t j = 0; j < src_tensor_shape[1]; ++j) {
        sigma2[i] += pow(p[j] - mu[i], 2);
      }
      sigma2[i] = sigma2[i] / src_tensor_shape[1];
    }
    // calculate dst
    for (int64_t i = 0; i < src_tensor_shape[0]; ++i) {
      for (int64_t j = 0; j < src_tensor_shape[1]; ++j) {
        auto src_sub_mu = src_tensor_data[i * src_strides[0] + j * src_strides[1]] - mu[i];
        dst_data[i * dst_strides[0] + j * dst_strides[1]] =
            (src_sub_mu / sqrt(sigma2[i] + epsilon)) * gamma_tensor_data[j] + beta_tensor_data[j];
      }
    }
  }
  delete[] mu;
  delete[] sigma2;
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::LayerNormOperator lnorm_op(p.conf);
    lnorm_op.Reshape(p.input, p.output);
    lnorm_op.Forward(p.input, p.output);
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
                                        q.output[0]->size(), 4e-6);
  }
  return false;
}

class LayerNormOpTest : public testing::TestWithParam<TestParams> {
 protected:
  LayerNormOpTest() {}
  ~LayerNormOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LayerNormOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape) {
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
  attr_map["epsilon"] = "0.0010000000474974513";
  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("layer_norm", "fp32", input_config_vec, output_config_vec, op_attr);

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
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> gamma_shape;
  std::vector<int64_t> beta_shape;

  // case: simple 2D
  src_shape = {16, 32};
  gamma_shape = {1, 32};
  beta_shape = {1, 32};
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}), false});
  // case: expect fail
  src_shape = {4, 4, 4, 8, 16, 32};
  gamma_shape = {1, 32};
  beta_shape = {1, 32};
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape, beta_shape}), true});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, LayerNormOpTest, CasesFp32());
