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

#include "reshape.hpp"

namespace executor {

ReshapeOperator::ReshapeOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("dst_shape");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&shape_, attrs_map["dst_shape"], ",");
  iter = attrs_map.find("dims");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&dims_, attrs_map["dims"], ",");
  iter = attrs_map.find("mul");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&mul_, attrs_map["mul"], ",");
}

ReshapeOperator::~ReshapeOperator() {}

void ReshapeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype(input[0]->dtype());
}

void ReshapeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Set dst tensor shape
  vector<int64_t> pre_dst_shape(shape_);
  if (input.size() == 2) {
    auto shape_vec = input[1]->shape();
    int j = 0;
    for (int i = 0; i < pre_dst_shape.size(); i++) {
      if (pre_dst_shape[i] == -1) {
        pre_dst_shape[i] = shape_vec[dims_[j++]];
      }
      if (j >= dims_.size()) {
        break;
      }
    }
  }

  Tensor* src_ptr = input[0];
  int64_t src_size = src_ptr->size();
  int idx = -1;
  int64_t shape_acc = 1;
  for (int i = 0; i < pre_dst_shape.size(); i++) {
    if (pre_dst_shape[i] != -1) {
      shape_acc *= pre_dst_shape[i];
    } else {
      idx = i;
    }
  }
  if (idx != -1) {
    pre_dst_shape[idx] = src_size / shape_acc;
  }
  vector<int64_t> dst_shape(pre_dst_shape);
  if (mul_.size() > 1) {
    dst_shape.clear();
    int j = 0;
    int64_t mul_size = 1;
    for (int i = 0; i < mul_.size(); ++i) mul_size *= pre_dst_shape[mul_[i]];
    for (int i = 0; i < pre_dst_shape.size(); ++i) {
      if (j < mul_.size() && i == mul_[j]) {
        if (j == 0) dst_shape.push_back(mul_size);
        j++;
      } else {
        dst_shape.push_back(pre_dst_shape[i]);
      }
    }
  }

  Tensor* dst_ptr = output[0];
  dst_ptr->set_shape(dst_shape);
}

// 2. inference kernel(for int8 and f32)
void ReshapeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src_ptr = input[0];
  Tensor* dst_ptr = output[0];
  auto data = src_ptr->mutable_data();
  auto life_count = MemoryAllocator::get().CheckMemory(data);

  // set data is inplace mamupulation, will reset so no need
  if (life_count == 1) {
    src_ptr->unref_data(true);
    dst_ptr->set_data(data);
    if (input.size() == 2) input[1]->unref_data();
  } else {
    void* dst_data_ptr = const_cast<void*>(dst_ptr->mutable_data());
    int data_size = dst_ptr->size();
    string data_type = src_ptr->dtype();
    memcpy(dst_data_ptr, data, data_size * type2bytes[data_type]);
    LOG(WARNING) << "input tensor" << src_ptr->name() << " will be used by multi node...";
    this->unref_tensors(input);
  }
}

REGISTER_OPERATOR_CLASS(Reshape);
}  // namespace executor
