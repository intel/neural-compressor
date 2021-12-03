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
#include "../../include/operators/matmul.hpp"
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
  vector<int64_t> src1_perm;
  executor::StringSplit<int64_t>(&src1_perm, attrs_map["src1_perm"], ",");
  vector<int64_t> dst_perm;
  executor::StringSplit<int64_t>(&dst_perm, attrs_map["dst_perm"], ",");
  bool dst_need_trans = (attrs_map["dst_perm"] != "" || attrs_map["dst_perm"] != "0,1,2,3");

  auto src0_tensor_shape = input[0]->shape();
  const auto src0_tensor_data = static_cast<const float*>(input[0]->data());
  auto src1_tensor_shape = input[1]->shape();
  const auto src1_tensor_data = static_cast<const float*>(input[1]->data());

  vector<int64_t> src0_shape = executor::GetShapes(src0_tensor_shape, src0_perm);
  vector<int64_t> src1_shape = executor::GetShapes(src1_tensor_shape, src1_perm);
  vector<int64_t> src0_stride = executor::GetStrides(src0_tensor_shape, src0_perm);
  vector<int64_t> src1_stride = executor::GetStrides(src1_tensor_shape, src1_perm);

  vector<int64_t> dst_shape_origin = src0_shape;
  dst_shape_origin.back() = src1_shape.back();
  vector<int64_t> dst_stride_origin = executor::GetStrides(dst_shape_origin);

  vector<int64_t> dst_shape_trans = executor::GetShapes(dst_shape_origin, dst_perm);
  vector<int64_t> dst_stride_trans = executor::GetStrides(dst_shape_origin, dst_perm);

  // dst shape
  output[0]->set_shape(dst_shape_trans);
  float* dst_data = static_cast<float*>(output[0]->mutable_data());

  // Execute test kernel
  int src0_dims = src0_shape.size();
  int first_idx = src0_dims - 2;
  int second_idx = src0_dims - 1;
  int BS0 = src0_dims == 4 ? dst_shape_origin[src0_dims - 4] : 1;
  int BS1 = src0_dims >= 3 ? dst_shape_origin[src0_dims - 3] : 1;
  int M = src0_shape[first_idx];
  int K = src0_shape[second_idx];
  int N = src1_shape[second_idx];

  if (src0_dims == 2) {
    for (int i = 0; i < M; ++i) {
// #pragma omp simd
#pragma omp parallel for
      for (int j = 0; j < N; ++j) {
        double value = 0;
#pragma omp simd
        for (int k = 0; k < K; ++k) {
          int src0_idx = i * src0_stride[0] + k * src0_stride[1];
          int src1_idx = k * src1_stride[0] + j * src1_stride[1];
          value += static_cast<double>(src0_tensor_data[src0_idx]) * static_cast<double>(src1_tensor_data[src1_idx]);
        }
        int dst_idx = i * dst_stride_origin[0] + j * dst_stride_origin[1];
        dst_data[dst_idx] = value;
      }
    }
    return;
  }
  if (src0_dims == 4) {
    float* dst_buffer = dst_need_trans ? new float[output[0]->size()] : dst_data;
    // Dst without transpose
    for (int bs0 = 0; bs0 < BS0; ++bs0) {
      for (int bs1 = 0; bs1 < BS1; ++bs1) {
        for (int i = 0; i < M; ++i) {
// #pragma omp simd
#pragma omp parallel for
          for (int j = 0; j < N; ++j) {
            double value = 0;
#pragma omp simd
            for (int k = 0; k < K; ++k) {
              int src0_idx = bs0 * +src0_stride[0] + bs1 * src0_stride[1] + i * src0_stride[2] + k * src0_stride[3];
              int src1_idx = bs0 * +src1_stride[0] + bs1 * src1_stride[1] + k * src1_stride[2] + j * src1_stride[3];
              value +=
                  static_cast<double>(src0_tensor_data[src0_idx]) * static_cast<double>(src1_tensor_data[src1_idx]);
            }
            int dst_idx = bs0 * +dst_stride_origin[0] + bs1 * dst_stride_origin[1] + i * dst_stride_origin[2] +
                          j * dst_stride_origin[3];
            dst_buffer[dst_idx] = value;
          }
        }
      }
    }
    // if dst_need_trans, do dst transpose
    if (dst_need_trans) {
      vector<int64_t> final_stride = executor::GetStrides(dst_shape_trans);
      for (int bs0 = 0; bs0 < dst_shape_trans[0]; ++bs0) {
        for (int bs1 = 0; bs1 < dst_shape_trans[1]; ++bs1) {
#pragma omp parallel for
          for (int i = 0; i < dst_shape_trans[2]; ++i) {
#pragma omp simd
            for (int j = 0; j < dst_shape_trans[3]; ++j) {
              int src_idx = bs0 * +dst_stride_trans[0] + bs1 * dst_stride_trans[1] + i * dst_stride_trans[2] +
                            j * dst_stride_trans[3];
              int data_idx = bs0 * +final_stride[0] + bs1 * final_stride[1] + i * final_stride[2] + j * final_stride[3];
              dst_data[data_idx] = dst_buffer[src_idx];
            }
          }
        }
      }
      delete[] dst_buffer;
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::MatmulOperator matmul(p.conf);
    matmul.Prepare(p.input, p.output);
    matmul.Reshape(p.input, p.output);
    matmul.Forward(p.input, p.output);
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
                                        q.output[0]->size(), 1e-3);
  }
  return false;
}

class MatmulTest : public testing::TestWithParam<TestParams> {
 protected:
  MatmulTest() {}
  ~MatmulTest() override {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MatmulTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           std::string src0_perm = "", std::string src1_perm = "",
                                           std::string dst_perm = "", std::string format_any = "false",
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  TensorConfig* src0_config = new TensorConfig("src0", src0_shape);
  TensorConfig* src1_config = new TensorConfig("src1", src1_shape);
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> inputs_config = {src0_config, src1_config};
  if (append_op == "sum") {
    inputs_config.push_back(new TensorConfig("src2", input_shape[2]));
  }

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"src0_perm", src0_perm},   {"src1_perm", src1_perm}, {"dst_perm", dst_perm},
              {"format_any", format_any}, {"output_dtype", "fp32"}, {"append_op", append_op}};

  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("matmul", "fp32", inputs_config, {dst_config}, op_attr);

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
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;

  // case: simple
  src0_shape = {1, 2};
  src1_shape = {2, 1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), false});
  // case: simple
  src0_shape = {10, 5};
  src1_shape = {5, 8};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), false});
  // case: format_tag::any
  src0_shape = {10, 5};
  src1_shape = {5, 8};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "0,1", "0,1", "0,1", "true"), false});
  // case: adj_y = true
  src0_shape = {10, 20};
  src1_shape = {32, 20};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "0,1", "1,0"), false});
  // case: tensor_x: perm={0, 2, 1, 3} + adj_x=false. tensor_y: perm={0, 2, 1, 3} + adj_y=true
  src0_shape = {2, 128, 12, 64};
  src1_shape = {2, 128, 12, 64};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "0,2,1,3", "0,2,3,1"), false});
  // case: tensor_x: perm={0, 1, 2, 3} + adj_x=false. tensor_y: perm={0, 2, 1, 3} + adj_y=false.
  // tensor_dst: perm={0, 2, 1, 3}
  src0_shape = {2, 12, 128, 128};
  src1_shape = {2, 128, 12, 64};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "", "0,2,1,3", "0,2,1,3"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, MatmulTest, CasesFp32());
