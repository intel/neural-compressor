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

#include "group_norm.hpp"

#include "common.hpp"

namespace executor {

void GroupNormRef(const float* src_data, const float* gamma_data, const float* beta_data, float* dst_data,
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

GroupNormOperator::GroupNormOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("epsilon");
  if (iter != attrs_map.end()) {
    epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
  }
  iter = attrs_map.find("group");
  if (iter != attrs_map.end()) {
    group_ = StringToNum<int64_t>(attrs_map["group"]);
  }
  iter = attrs_map.find("channels");
  if (iter != attrs_map.end()) {
    channels_ = StringToNum<int64_t>(attrs_map["channels"]);
  }
}

void GroupNormOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  assert(src->dtype() == "fp32");
  output[0]->set_dtype(src->dtype());
  Tensor* gamma = input[1];
  Tensor* beta = input[2];
  assert(gamma->shape()[0] == channels_);
  assert(beta->shape()[0] == channels_);
  const float* gamma_data = static_cast<const float*>(gamma->data());
  const float* beta_data = static_cast<const float*>(beta->data());
  for (int64_t i = 0; i < channels_; ++i) {
    if (gamma_data[i] != 1.f || beta_data[i] != 0.f) {
      affine_ = true;
      break;
    }
  }
}

void GroupNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> src_shape = input[0]->shape();
  assert(src_shape.size() > 1);
  output[0]->set_shape(src_shape);
}

void GroupNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  const vector<int64_t>& src_shape = src->shape();
  const float* src_data = static_cast<const float*>(src->data());
  Tensor* gamma = input[1];
  const float* gamma_data = static_cast<const float*>(gamma->data());
  Tensor* beta = input[2];
  const float* beta_data = static_cast<const float*>(beta->data());
  Tensor* dst = output[0];
  float* dst_data = static_cast<float*>(dst->mutable_data());
  GroupNormRef(src_data, gamma_data, beta_data, dst_data, src_shape, epsilon_, group_, channels_, affine_);

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(GroupNorm);
}  // namespace executor
