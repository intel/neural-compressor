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

#include "slice.hpp"

#include "common.hpp"

namespace executor {

template <typename T>
void SliceData(const T* src_data, T* dst_data, const vector<int64_t>& src_shape, const vector<int64_t>& dst_shape,
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
template void SliceData<float>(const float* src_data, float* dst_data, const vector<int64_t>& src_shape,
                               const vector<int64_t>& dst_shape, const vector<int64_t>& starts,
                               const vector<int64_t>& ends, const vector<int64_t>& axes, const vector<int64_t>& steps);
template void SliceData<int32_t>(const int32_t* src_data, int32_t* dst_data, const vector<int64_t>& src_shape,
                                 const vector<int64_t>& dst_shape, const vector<int64_t>& starts,
                                 const vector<int64_t>& ends, const vector<int64_t>& axes,
                                 const vector<int64_t>& steps);
template void SliceData<uint16_t>(const uint16_t* src_data, uint16_t* dst_data, const vector<int64_t>& src_shape,
                                  const vector<int64_t>& dst_shape, const vector<int64_t>& starts,
                                  const vector<int64_t>& ends, const vector<int64_t>& axes,
                                  const vector<int64_t>& steps);
template void SliceData<uint8_t>(const uint8_t* src_data, uint8_t* dst_data, const vector<int64_t>& src_shape,
                                 const vector<int64_t>& dst_shape, const vector<int64_t>& starts,
                                 const vector<int64_t>& ends, const vector<int64_t>& axes,
                                 const vector<int64_t>& steps);
template void SliceData<int8_t>(const int8_t* src_data, int8_t* dst_data, const vector<int64_t>& src_shape,
                                const vector<int64_t>& dst_shape, const vector<int64_t>& starts,
                                const vector<int64_t>& ends, const vector<int64_t>& axes, const vector<int64_t>& steps);

SliceOperator::SliceOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("starts");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&starts_, attrs_map["starts"], ",");
  }
  iter = attrs_map.find("ends");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&ends_, attrs_map["ends"], ",");
  }
  iter = attrs_map.find("axes");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&axes_, attrs_map["axes"], ",");
  }
  iter = attrs_map.find("steps");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&steps_, attrs_map["steps"], ",");
  }
}

void SliceOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype(input[0]->dtype());
}

void SliceOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t>& src_shape = input[0]->shape();
  vector<int64_t> dst_shape = src_shape;
  for (int64_t i = 0; i < axes_.size(); ++i) {
    starts_[i] = starts_[i] < 0 ? src_shape[axes_[i]] + starts_[i] : starts_[i];
    ends_[i] = ends_[i] < 0 ? src_shape[axes_[i]] + ends_[i] : ends_[i];
    dst_shape[axes_[i]] = static_cast<int64_t>((ends_[i] - starts_[i]) / steps_[i]) + 1;
  }
  output[0]->set_shape(dst_shape);
}

void SliceOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  Tensor* dst = output[0];
  const vector<int64_t>& src_shape = src->shape();
  const vector<int64_t>& dst_shape = dst->shape();
  if (src->dtype() == "fp32") {
    const float* src_data = static_cast<const float*>(src->data());
    float* dst_data = static_cast<float*>(dst->mutable_data());
    SliceData<float>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "s32") {
    const int32_t* src_data = static_cast<const int32_t*>(src->data());
    int32_t* dst_data = static_cast<int32_t*>(dst->mutable_data());
    SliceData<int32_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "bf16") {
    const uint16_t* src_data = static_cast<const uint16_t*>(src->data());
    uint16_t* dst_data = static_cast<uint16_t*>(dst->mutable_data());
    SliceData<uint16_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "u8") {
    const uint8_t* src_data = static_cast<const uint8_t*>(src->data());
    uint8_t* dst_data = static_cast<uint8_t*>(dst->mutable_data());
    SliceData<uint8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else if (src->dtype() == "s8") {
    const int8_t* src_data = static_cast<const int8_t*>(src->data());
    int8_t* dst_data = static_cast<int8_t*>(dst->mutable_data());
    SliceData<int8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
  } else {
    LOG(ERROR) << "Dtype " << src->dtype() << "is not supported in slice op!";
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Slice);
}  // namespace executor
