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
#include "../../include/operators/convolution.hpp"
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

void Conv2D(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  auto attrs_map = conf.attributes();
  vector<int64_t> src_perm;
  executor::StringSplit<int64_t>(&src_perm, attrs_map["src_perm"], ",");
  if (src_perm.empty()) {
    src_perm = {0, 1, 2, 3};
  }
  vector<int64_t> dst_perm;
  executor::StringSplit<int64_t>(&dst_perm, attrs_map["dst_perm"], ",");
  if (dst_perm.empty()) {
    dst_perm = {0, 1, 2, 3};
  }
  int64_t group = 1;
  group = executor::StringToNum<int64_t>(attrs_map["group"]);
  vector<int64_t> pads;
  executor::StringSplit<int64_t>(&pads, attrs_map["pads"], ",");
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  vector<int64_t> strides;
  executor::StringSplit<int64_t>(&strides, attrs_map["strides"], ",");
  if (strides.empty()) {
    strides = {0, 0, 0, 0};
  }
  auto iter = attrs_map.find("append_op");
  if (iter != attrs_map.end()) {
    LOG(INFO) << "The append op of convolution is: " << iter->second;
  }
  bool relu = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  bool binary_add = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  bool gelu_erf = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  bool tanh = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  bool gelu_tanh = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;

  Tensor* src = input[0];
  Tensor* weight = input[1];
  Tensor* bias = input[2];
  const float* src_tensor_data = static_cast<const float*>(src->data());
  const float* wei_tensor_data = static_cast<const float*>(weight->data());
  const float* bias_tensor_data = static_cast<const float*>(bias->data());
  const float* post_tensor_data = binary_add ? static_cast<const float*>(input[3]->data()) : nullptr;

  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src_shape_origin = src->shape();
  vector<int64_t> src_shape = executor::GetShapes(src_shape_origin, src_perm);
  vector<int64_t> src_stride = executor::GetStrides(src_shape_origin, src_perm);
  vector<int64_t> weight_shape_origin = weight->shape();
  vector<int64_t> weight_shape = executor::GetShapes(weight_shape_origin);
  vector<int64_t> weight_stride = executor::GetStrides(weight_shape_origin);

  // 1.2 malloc tensor for output
  // src_: N * IC* IH * IW, weight_: OC * IC * KH * KW
  // pad: (PH_L, PH_R, PW_L, PW_R), stride: (SH, SW)
  // OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
  // OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width
  // dst_: N * OC * OH * OW
  const int64_t N = src_shape[0];
  const int64_t IC = src_shape[1];
  const int64_t IH = src_shape[2];
  const int64_t IW = src_shape[3];
  const int64_t OC = weight_shape[0];
  const int64_t KC = weight_shape[1];
  const int64_t KH = weight_shape[2];
  const int64_t KW = weight_shape[3];
  const int64_t PH_L = pads[0];
  const int64_t PH_R = pads[1];
  const int64_t PW_L = pads[2];
  const int64_t PW_R = pads[3];
  const int64_t SH = strides[0];
  const int64_t SW = strides[1];
  const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
  const int64_t OW = (IW - KW + PW_L + PW_R) / SW + 1;
  const int64_t IPH = IH + PH_L + PH_R;
  const int64_t IPW = IW + PW_L + PW_R;
  vector<int64_t> dst_shape_origin = {N, OC, OH, OW};
  output[0]->set_shape(dst_shape_origin);  // dst shape
  vector<int64_t> output_shape = executor::GetShapes(dst_shape_origin, dst_perm);
  vector<int64_t> output_stride = executor::GetStrides(dst_shape_origin, dst_perm);

  const int64_t input_pad_size = N * IC * IPH * IPW;
  vector<int64_t> pad_shape_origin = {N, IC, IPH, IPW};
  vector<int64_t> pad_stride = executor::GetStrides(pad_shape_origin, src_perm);
  float* src_pad_data = reinterpret_cast<float*>(malloc(input_pad_size * sizeof(float)));
  memset(src_pad_data, 0, input_pad_size * sizeof(float));
  for (int n = 0; n < N; n++) {
    for (int ic = 0; ic < IC; ic++) {
      for (int iph = PH_L; iph < PH_L + IH; iph++) {
        const int64_t origin_stride = n * src_stride[0] + ic * src_stride[1] + (iph - PH_L) * src_stride[2];
        const int64_t input_pad_stride = n * pad_stride[0] + ic * pad_stride[1] + iph * pad_stride[2];
        memcpy(src_pad_data + input_pad_stride + PW_L, src_tensor_data + origin_stride, IW * sizeof(float));
      }
    }
  }

  const int64_t num_output_g = OC / group;
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
#pragma omp simd
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < num_output_g; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
          for (int ow = 0; ow < OW; ++ow) {
            float sum_data = bias_tensor_data[g * num_output_g + oc];
            for (int kc = 0; kc < KC; ++kc) {
              float* pad_map_data =
                  src_pad_data + n * pad_stride[0] + (g * KC + kc) * pad_stride[1] + oh * SH * pad_stride[2] + ow * SW;
              for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                  sum_data += wei_tensor_data[(g * num_output_g + oc) * weight_stride[0] + kc * weight_stride[1] +
                                              kh * weight_stride[2] + kw] *
                              pad_map_data[kh * pad_stride[2] + kw];
                }
              }
            }
            if (gelu_erf) {
              const float sqrt_2_over_2 = 0.707106;
              float v = sum_data * sqrt_2_over_2;
              sum_data = (sqrt_2_over_2 * v * (1.f + ::erff(v)));
            } else if (gelu_tanh) {
              const float a = 0.797884;
              const float b = 0.044715;
              const float g = a * sum_data * (1 + b * sum_data * sum_data);
              sum_data = 0.5 * sum_data * (1 + ::tanhf(g));
            } else if (relu) {
              sum_data = sum_data < 0 ? 0 : sum_data;
            } else if (binary_add) {
              sum_data += post_tensor_data[n * output_stride[0] + (g * num_output_g + oc) * output_stride[1] +
                                           oh * output_stride[2] + ow];
            } else if (tanh) {
              sum_data = tanhf(sum_data);
            }
            dst_data[n * output_stride[0] + (g * num_output_g + oc) * output_stride[1] + oh * output_stride[2] + ow] =
                sum_data;
          }
        }
      }
    }
  }
  free(src_pad_data);
}

void Conv1D(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  auto attrs_map = conf.attributes();
  vector<int64_t> src_perm;
  executor::StringSplit<int64_t>(&src_perm, attrs_map["src_perm"], ",");
  if (src_perm.empty()) {
    src_perm = {0, 1, 2};
  }
  vector<int64_t> dst_perm;
  executor::StringSplit<int64_t>(&dst_perm, attrs_map["dst_perm"], ",");
  if (dst_perm.empty()) {
    dst_perm = {0, 1, 2};
  }
  int64_t group = 1;
  group = executor::StringToNum<int64_t>(attrs_map["group"]);
  vector<int64_t> pads;
  executor::StringSplit<int64_t>(&pads, attrs_map["pads"], ",");
  if (pads.empty()) {
    pads = {0, 0, 0};
  }
  vector<int64_t> strides;
  executor::StringSplit<int64_t>(&strides, attrs_map["strides"], ",");
  if (strides.empty()) {
    strides = {0, 0, 0};
  }
  auto iter = attrs_map.find("append_op");
  bool relu = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  bool binary_add = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  bool gelu_erf = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  bool tanh = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  bool gelu_tanh = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;

  Tensor* src = input[0];
  Tensor* weight = input[1];
  Tensor* bias = input[2];
  const float* src_tensor_data = static_cast<const float*>(src->data());
  const float* wei_tensor_data = static_cast<const float*>(weight->data());
  const float* bias_tensor_data = static_cast<const float*>(bias->data());
  const float* post_tensor_data = binary_add ? static_cast<const float*>(input[3]->data()) : nullptr;

  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src_shape_origin = src->shape();
  vector<int64_t> src_shape = executor::GetShapes(src_shape_origin, src_perm);
  vector<int64_t> src_stride = executor::GetStrides(src_shape_origin, src_perm);
  vector<int64_t> weight_shape_origin = weight->shape();
  vector<int64_t> weight_shape = executor::GetShapes(weight_shape_origin);
  vector<int64_t> weight_stride = executor::GetStrides(weight_shape_origin);

  // 1.2 malloc tensor for output
  // src_: N * IC* IH * IW, weight_: OC * IC * KH * KW
  // pad: (PH_L, PH_R, PW_L, PW_R), stride: (SH, SW)
  // OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
  // dst_: N * OC * OH * OW
  const int64_t N = src_shape[0];
  const int64_t IC = src_shape[1];
  const int64_t IH = src_shape[2];
  const int64_t OC = weight_shape[0];
  const int64_t KC = weight_shape[1];
  const int64_t KH = weight_shape[2];
  const int64_t PH_L = pads[0];
  const int64_t PH_R = pads[1];
  const int64_t SH = strides[0];
  const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
  const int64_t IPH = IH + PH_L + PH_R;
  vector<int64_t> dst_shape_origin = {N, OC, OH};
  output[0]->set_shape(dst_shape_origin);  // dst shape
  vector<int64_t> output_shape = executor::GetShapes(dst_shape_origin, dst_perm);
  vector<int64_t> output_stride = executor::GetStrides(dst_shape_origin, dst_perm);

  const int64_t input_pad_size = N * IC * IPH;
  vector<int64_t> pad_shape_origin = {N, IC, IPH};
  vector<int64_t> pad_stride = executor::GetStrides(pad_shape_origin, src_perm);
  float* src_pad_data = reinterpret_cast<float*>(malloc(input_pad_size * sizeof(float)));
  memset(src_pad_data, 0, input_pad_size * sizeof(float));
  for (int n = 0; n < N; n++) {
    for (int ic = 0; ic < IC; ic++) {
      const int64_t origin_stride = n * src_stride[0] + ic * src_stride[1];
      const int64_t input_pad_stride = n * pad_stride[0] + ic * pad_stride[1];
      memcpy(src_pad_data + input_pad_stride + PH_L, src_tensor_data + origin_stride, IH * sizeof(float));
    }
  }

  const int64_t num_output_g = OC / group;
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
#pragma omp simd
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < num_output_g; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
          float sum_data = bias_tensor_data[g * num_output_g + oc];
          for (int kc = 0; kc < KC; ++kc) {
            float* pad_map_data = src_pad_data + n * pad_stride[0] + (g * KC + kc) * pad_stride[1] + oh * SH;
            for (int kh = 0; kh < KH; ++kh) {
              sum_data += wei_tensor_data[(g * num_output_g + oc) * weight_stride[0] + kc * weight_stride[1] + kh] *
                          pad_map_data[kh];
            }
          }
          if (gelu_erf) {
            const float sqrt_2_over_2 = 0.707106;
            float v = sum_data * sqrt_2_over_2;
            sum_data = (sqrt_2_over_2 * v * (1.f + ::erff(v)));
          } else if (gelu_tanh) {
            const float a = 0.797884;
            const float b = 0.044715;
            const float g = a * sum_data * (1 + b * sum_data * sum_data);
            sum_data = 0.5 * sum_data * (1 + ::tanhf(g));
          } else if (relu) {
            sum_data = sum_data < 0 ? 0 : sum_data;
          } else if (binary_add) {
            sum_data += post_tensor_data[n * output_stride[0] + (g * num_output_g + oc) * output_stride[1] + oh];
          } else if (tanh) {
            sum_data = tanhf(sum_data);
          }
          dst_data[n * output_stride[0] + (g * num_output_g + oc) * output_stride[1] + oh] = sum_data;
        }
      }
    }
  }

  free(src_pad_data);
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output, const OperatorConfig& conf) {
  Tensor* src = input[0];
  vector<int64_t> src_shape = src->shape();
  switch (src_shape.size()) {
    case 3:
      Conv1D(input, output, conf);
      break;
    case 4:
      Conv2D(input, output, conf);
      break;
    default:
      LOG(ERROR) << "Input size " << src_shape.size() << " is not supported in convolution!";
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::ConvolutionOperator convolution(p.conf);
    convolution.Prepare(p.input, p.output);
    convolution.Reshape(p.input, p.output);
    convolution.Forward(p.input, p.output);
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
                                        q.output[0]->size(), 5e-3);
  }
  return false;
}

class InnerProductTest : public testing::TestWithParam<TestParams> {
 protected:
  InnerProductTest() {}
  ~InnerProductTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(InnerProductTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           const std::string& src_perm, const std::string& dst_perm,
                                           const std::string& group, const std::string& pads,
                                           const std::string& strides, const std::string& output_dtype,
                                           const std::string& append_op) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  const auto& weight_shape = input_shape[1];
  const auto& bias_shape = input_shape[2];
  TensorConfig* src_config = new TensorConfig("src", src_shape);
  TensorConfig* weight_config = new TensorConfig("weight", weight_shape);
  TensorConfig* bias_config = new TensorConfig("bias", bias_shape);
  std::vector<int64_t> dst_shape = {};
  TensorConfig* dst_config = new TensorConfig("dst", dst_shape);
  std::vector<TensorConfig*> inputs_config = {src_config, weight_config, bias_config};
  if (append_op == "binary_add") {
    inputs_config.push_back(new TensorConfig("src2", input_shape[3]));
  }

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"src_perm", src_perm}, {"dst_perm", dst_perm},         {"group", group},        {"pads", pads},
              {"strides", strides},   {"output_dtype", output_dtype}, {"append_op", append_op}};

  AttrConfig* op_attr = new AttrConfig(attr_map);
  OperatorConfig op_config = OperatorConfig("convolution", output_dtype, inputs_config, {dst_config}, op_attr);

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
  auto weight_tensors = make_tensor_obj(weight_config);
  auto bias_tensors = make_tensor_obj(bias_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, weight_tensors.first, bias_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {
      {src_tensors.second, weight_tensors.second, bias_tensors.second}, {dst_tensor_copy}, op_config};

  if (append_op == "binary_add") {
    auto src2_tensors = make_tensor_obj(inputs_config[3]);
    op_args.input.push_back(src2_tensors.first);
    op_args_copy.input.push_back(src2_tensors.second);
  }
  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> bias_shape;
  std::vector<int64_t> post_shape;
  std::string group = "";
  std::string pads = "";
  std::string strides = "";
  std::string src_perm = "";
  std::string dst_perm = "";
  std::string output_dtype = "fp32";
  std::string append_op = "";

  // case1: 2d conv
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});
  // case2: 2d conv, gelu_erf
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "gelu_erf";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});
  // case3: 2d conv, gelu_tanh
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "gelu_tanh";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case4: 2d conv, relu
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "relu";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case5: 2d conv, tanh
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "tanh";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case6: 2d conv, binary_add
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 32, 3, 3};
  bias_shape = {64};
  post_shape = {3, 64, 4, 4};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "binary_add";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape, post_shape}, src_perm, dst_perm, group, pads,
                                    strides, output_dtype, append_op),
                   false});

  // case7: 2d conv, binary_add, group=2
  src_shape = {3, 32, 13, 13};
  weight_shape = {64, 16, 3, 3};
  bias_shape = {64};
  post_shape = {3, 64, 4, 4};
  src_perm = "";
  dst_perm = "";
  group = "2";
  pads = "1,1,1,1";
  strides = "4,4";
  output_dtype = "fp32";
  append_op = "binary_add";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape, post_shape}, src_perm, dst_perm, group, pads,
                                    strides, output_dtype, append_op),
                   false});

  // case8: 1d conv
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case9: 1d conv, gelu_erf
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "gelu_erf";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case10: 1d conv, gelu_tanh
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "gelu_tanh";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case11: 1d conv, relu
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "relu";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case12: 1d conv, relu
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "relu";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case13: 1d conv, tanh
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "tanh";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case14: 1d conv, binary_add
  src_shape = {3, 32, 13};
  weight_shape = {64, 32, 3};
  bias_shape = {64};
  post_shape = {3, 64, 4};
  src_perm = "0,1,2";
  dst_perm = "0,1,2";
  group = "1";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "binary_add";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape, post_shape}, src_perm, dst_perm, group, pads,
                                    strides, output_dtype, append_op),
                   false});

  // case15: 1d conv, group=2
  src_shape = {3, 32, 13};
  weight_shape = {64, 16, 3};
  bias_shape = {64};
  src_perm = "";
  dst_perm = "";
  group = "2";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape}, src_perm, dst_perm, group, pads, strides,
                                    output_dtype, append_op),
                   false});

  // case16: 1d conv, group=2, binary_add
  src_shape = {3, 32, 13};
  weight_shape = {64, 16, 3};
  bias_shape = {64};
  post_shape = {3, 64, 4};
  src_perm = "";
  dst_perm = "";
  group = "2";
  pads = "1,1";
  strides = "4";
  output_dtype = "fp32";
  append_op = "binary_add";

  cases.push_back({GenerateFp32Case({src_shape, weight_shape, bias_shape, post_shape}, src_perm, dst_perm, group, pads,
                                    strides, output_dtype, append_op),
                   false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, InnerProductTest, CasesFp32());
