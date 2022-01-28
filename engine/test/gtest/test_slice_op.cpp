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
#include "../../include/operators/slice.hpp"
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

template <typename T>
void SliceDataGT(const T* src_data, T* dst_data, const vector<int64_t>& src_shape, const vector<int64_t>& dst_shape,
                 const vector<int64_t>& starts, const vector<int64_t>& ends, const vector<int64_t>& axes,
                 const vector<int64_t>& steps) {
  int64_t src_size = 1;
  for (auto shape : src_shape) {
    src_size *= shape;
  }
  T* src_data_tmp = static_cast<T*>(malloc(src_size * sizeof(T)));
  T* dst_data_tmp = static_cast<T*>(malloc(src_size * sizeof(T)));
  memcpy(src_data_tmp, src_data, src_size * sizeof(T));
  memcpy(dst_data_tmp, src_data, src_size * sizeof(T));
  vector<int64_t> src_shape_tmp = src_shape;
  vector<int64_t> dst_shape_tmp = src_shape;
  for (int64_t i = 0; i < axes.size(); ++i) {
    dst_shape_tmp[axes[i]] = static_cast<int64_t>((ends[i] - starts[i]) / steps[i]) + 1;
    int64_t IN = 1;
    int64_t IC = 1;
    int64_t IH = 1;
    int64_t ON = 1;
    int64_t OC = 1;
    int64_t OH = 1;
    int64_t step = steps[i];
    for (int64_t j = 0; j < axes[i]; ++j) {
      IN *= src_shape_tmp[j];
      ON *= dst_shape_tmp[j];
    }
    IC = src_shape_tmp[axes[i]];
    OC = dst_shape_tmp[axes[i]];
    for (int64_t j = axes[i] + 1; j < src_shape_tmp.size(); ++j) {
      IH *= src_shape_tmp[j];
      OH *= dst_shape_tmp[j];
    }
    int64_t start = starts[i] * IH;
#pragma omp parallel for
    for (int64_t on = 0; on < ON; ++on) {
#pragma omp simd
      for (int64_t oc = 0; oc < OC; ++oc) {
        memcpy(dst_data_tmp + on * OC * OH + oc * OH, src_data_tmp + start + on * IC * IH + (oc * step) * IH,
               OH * sizeof(T));
      }
    }
    memcpy(src_data_tmp, dst_data_tmp, ON * OC * OH * sizeof(T));
    src_shape_tmp = dst_shape_tmp;
  }
  int64_t dst_size = 1;
  for (auto shape : dst_shape) {
    dst_size *= shape;
  }
  memcpy(dst_data, dst_data_tmp, dst_size * sizeof(T));
  free(src_data_tmp);
  free(dst_data_tmp);
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  // config parse
  auto attrs_map = conf.attributes();
  vector<int64_t> starts_;
  vector<int64_t> ends_;
  vector<int64_t> axes_;
  vector<int64_t> steps_;
  auto iter = attrs_map.find("starts");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&starts_, attrs_map["starts"], ",");
  }
  iter = attrs_map.find("ends");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&ends_, attrs_map["ends"], ",");
  }
  iter = attrs_map.find("axes");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&axes_, attrs_map["axes"], ",");
  }
  iter = attrs_map.find("steps");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&steps_, attrs_map["steps"], ",");
  }
  Tensor* src = input[0];
  Tensor* dst = output[0];
  // set dst dtype
  dst->set_dtype(src->dtype());
  // set dst shape
  const vector<int64_t>& src_shape = src->shape();
  vector<int64_t> dst_shape = src_shape;
  for (int64_t i = 0; i < axes_.size(); ++i) {
    starts_[i] = starts_[i] < 0 ? src_shape[axes_[i]] + starts_[i] : starts_[i];
    ends_[i] = ends_[i] < 0 ? src_shape[axes_[i]] + ends_[i] : ends_[i];
    dst_shape[axes_[i]] = static_cast<int64_t>((ends_[i] - starts_[i]) / steps_[i]) + 1;
  }
  output[0]->set_shape(dst_shape);
  // forward
  if (src->dtype() == "fp32") {
    const float* src_data = static_cast<const float*>(src->data());
    float* dst_data = static_cast<float*>(dst->mutable_data());
    SliceDataGT<float>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "s32") {
    const int32_t* src_data = static_cast<const int32_t*>(src->data());
    int32_t* dst_data = static_cast<int32_t*>(dst->mutable_data());
    SliceDataGT<int32_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "bf16") {
    const uint16_t* src_data = static_cast<const uint16_t*>(src->data());
    uint16_t* dst_data = static_cast<uint16_t*>(dst->mutable_data());
    SliceDataGT<uint16_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "u8") {
    const uint8_t* src_data = static_cast<const uint8_t*>(src->data());
    uint8_t* dst_data = static_cast<uint8_t*>(dst->mutable_data());
    SliceDataGT<uint8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "s8") {
    const int8_t* src_data = static_cast<const int8_t*>(src->data());
    int8_t* dst_data = static_cast<int8_t*>(dst->mutable_data());
    SliceDataGT<int8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else {
    LOG(ERROR) << "Dtype " << src->dtype() << "is not supported in slice op!";
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  executor::SliceOperator slice(p.conf);
  slice.Reshape(p.input, p.output);
  slice.Forward(p.input, p.output);

  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size());
  }
  return false;
}

class SliceTest : public testing::TestWithParam<TestParams> {
 protected:
  SliceTest() {}
  ~SliceTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SliceTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           const std::string& starts, const std::string& ends, const std::string& axes,
                                           const std::string& steps) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  std::vector<TensorConfig*> input_config = {src_config};
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> output_config = {dst_config, dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"axes", axes}, {"starts", starts}, {"ends", ends}, {"steps", steps}};

  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("slice", "fp32", input_config, output_config, op_attr);

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
  std::string starts;
  std::string ends;
  std::string axes;
  std::string steps;

  // case: 1d slice
  src_shape = {10};
  starts = "3";
  ends = "8";
  axes = "0";
  steps = "2";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 1d slice, ends=-1
  src_shape = {10};
  starts = "0";
  ends = "-1";
  axes = "0";
  steps = "3";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 1d slice, ends=-2
  src_shape = {10};
  starts = "0";
  ends = "-2";
  axes = "0";
  steps = "3";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 1d slice, start=end
  src_shape = {10};
  starts = "5";
  ends = "-4";
  axes = "0";
  steps = "3";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 2d slice, axes=0
  src_shape = {3, 2};
  starts = "1";
  ends = "-1";
  axes = "0";
  steps = "1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 2d slice, axes=1
  src_shape = {3, 2};
  starts = "1";
  ends = "-1";
  axes = "1";
  steps = "1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=0
  src_shape = {3, 2, 3};
  starts = "1";
  ends = "-1";
  axes = "0";
  steps = "1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=1
  src_shape = {3, 2, 3};
  starts = "1";
  ends = "-1";
  axes = "1";
  steps = "1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=2
  src_shape = {3, 2, 3};
  starts = "1";
  ends = "-1";
  axes = "2";
  steps = "1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=1,2
  src_shape = {3, 2, 3};
  starts = "1,2";
  ends = "-1,-1";
  axes = "1,2";
  steps = "1,1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=0,2
  src_shape = {3, 2, 3};
  starts = "1,2";
  ends = "-1,-1";
  axes = "0,2";
  steps = "1,1";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  // case: 3d slice, axes=0,2，step=2
  src_shape = {4, 2, 4};
  starts = "0,0";
  ends = "-1,-1";
  axes = "0,2";
  steps = "2,2";
  cases.push_back({GenerateFp32Case({src_shape}, starts, ends, axes, steps), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SliceTest, CasesFp32());
