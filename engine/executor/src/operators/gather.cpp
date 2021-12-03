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

#include "gather.hpp"
#if __AVX512F__
#include <immintrin.h>
#endif
#include "common.hpp"

namespace executor {

GatherOperator::GatherOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
  iter = attrs_map.find("batch_dims");
  batch_dims_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
}

void GatherOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const auto& indices_shape = input[0]->shape();
  const auto& params_shape = input[1]->shape();  // frozen data

  // 1.2 Get tensor's adjusted shapes
  // dst shape = params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis + 1:].
  vector<int64_t> dst_shape;
  for (int i = 0; i < axis_; ++i) {
    dst_shape.push_back(params_shape[i]);
  }
  for (int i = batch_dims_; i < indices_shape.size(); ++i) {
    dst_shape.push_back(indices_shape[i]);
  }
  for (int i = axis_ + 1; i < params_shape.size(); ++i) {
    dst_shape.push_back(params_shape[i]);
  }

  // 1.3 Get tensor's adjusted strides (cached)
  int64_t batch_size = 1;
  for (int i = 0; i < batch_dims_; ++i) {
    batch_size *= params_shape[i];
  }
  int64_t outer_size = 1;
  for (int i = batch_dims_; i < axis_; ++i) {
    outer_size *= params_shape[i];
  }
  int64_t coord_size = 1;
  for (int i = batch_dims_; i < indices_shape.size(); ++i) {
    coord_size *= indices_shape[i];
  }
  int64_t inner_size = 1;
  for (int i = axis_ + 1; i < params_shape.size(); ++i) {
    inner_size *= params_shape[i];
  }
  int64_t axis_size = params_shape[axis_];
  vector<int64_t> flat_params_shape = {batch_size, outer_size, axis_size, inner_size};
  flat_params_stride_ = GetStrides(flat_params_shape);
  flat_dst_shape_ = {batch_size, outer_size, coord_size, inner_size};
  flat_dst_stride_ = GetStrides(flat_dst_shape_);

  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);
}

void GatherOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& indices_data = static_cast<const int32_t*>(input[0]->data());
  const auto& params_data = static_cast<const int32_t*>(input[1]->data());  // frozen data
  // when change data value please use mutable_data
  auto dst_data = static_cast<float*>(output[0]->mutable_data());
  LOG_IF(ERROR, reinterpret_cast<void*>(dst_data) == reinterpret_cast<void*>(const_cast<int32_t*>(indices_data)))
      << "DST ptr should not be equal to SRC ptr.";

  // 1. Execute the dst
  const auto& batch_size = flat_dst_shape_[0];
  const auto& outer_size = flat_dst_shape_[1];
  const auto& coord_size = flat_dst_shape_[2];
  const auto& inner_size = flat_dst_shape_[3];
#if __AVX512F__
  int avx512_loop_len = inner_size >> 4;
  for (int i = 0; i < batch_size; ++i) {
    int indices_batch = i * coord_size;
    for (int j = 0; j < outer_size; ++j) {
#pragma omp parallel for
      for (int k = 0; k < coord_size; ++k) {
        int indices_val = indices_data[indices_batch + k];
        // copy slices on inner_size dimension
        for (int m = 0; m < avx512_loop_len; ++m) {
          int dst_idx = i * flat_dst_stride_[0] + j * flat_dst_stride_[1] + k * flat_dst_stride_[2] + (m << 4);
          int params_idx =
              i * flat_params_stride_[0] + j * flat_params_stride_[1] + indices_val * flat_params_stride_[2] + (m << 4);
          __m512i _src_data = _mm512_loadu_si512(params_data + params_idx);
          _mm512_storeu_si512(dst_data + dst_idx, _src_data);
        }
#pragma omp simd
        for (int tail_idx = avx512_loop_len << 4; tail_idx < inner_size; ++tail_idx) {
          int dst_idx = i * flat_dst_stride_[0] + j * flat_dst_stride_[1] + k * flat_dst_stride_[2] + tail_idx;
          int params_idx =
              i * flat_params_stride_[0] + j * flat_params_stride_[1] + indices_val * flat_params_stride_[2] + tail_idx;
          // for the both fp32 and int32 type data
          memcpy(dst_data + dst_idx, params_data + params_idx, 4);
        }
      }
    }
  }
#else
  for (int i = 0; i < batch_size; ++i) {
    int indices_batch = i * coord_size;
    for (int j = 0; j < outer_size; ++j) {
#pragma omp parallel for
      for (int k = 0; k < coord_size; ++k) {
        int indices_val = indices_data[indices_batch + k];
// copy slices on inner_size dimension
#pragma omp simd
        for (int m = 0; m < inner_size; ++m) {
          int dst_idx = i * flat_dst_stride_[0] + j * flat_dst_stride_[1] + k * flat_dst_stride_[2] + m;
          int params_idx =
              i * flat_params_stride_[0] + j * flat_params_stride_[1] + indices_val * flat_params_stride_[2] + m;
          // for the both fp32 and int32 type data
          memcpy(dst_data + dst_idx, params_data + params_idx, 4);
        }
      }
    }
  }
#endif
  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Gather);
}  // namespace executor
