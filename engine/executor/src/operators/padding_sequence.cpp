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

#include "padding_sequence.hpp"

#include "common.hpp"

namespace executor {

PaddingSequenceOperator::PaddingSequenceOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("padding_value");
  padding_ = (iter != attrs_map.end() && iter->second != "") ? StringToNum<float>(iter->second) : -10000;
  StringSplit<int64_t>(&attr_dst_shape_, attrs_map["dst_shape"], ",");
  if (attr_dst_shape_.empty()) {
    LOG(ERROR) << "dst_shape attr is empty.";
  }
  StringSplit<int64_t>(&dims_, attrs_map["dims"], ",");
}

void PaddingSequenceOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const auto& src0_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  int padding_idx = 0;
  int broadcast_idx = 0;
  vector<int64_t> dst_shape = attr_dst_shape_;
  for (int i = 0; i < attr_dst_shape_.size(); ++i) {
    if (attr_dst_shape_[i] == -1 && padding_idx < src0_shape.size()) {
      dst_shape[i] = src0_shape[padding_idx++];
      continue;
    }
    if (attr_dst_shape_[i] == 0 && broadcast_idx < dims_.size()) {
      dst_shape[i] = src0_shape[dims_[broadcast_idx++]];
      continue;
    }
  }

  // 1.3 Get tensor's adjusted strides (cached)
  src_shape_ = src0_shape;  // (batch_size, sequence_len)
  src_stride_ = GetStrides(src0_shape);

  LOG_IF(ERROR, dst_shape.size() < 2) << "Padding Sequence dst dims should be greater than 1.";
  int64_t bs = dst_shape[0];
  int64_t seq = dst_shape.back();
  // pad_dst_shape_ = {bs, broadcast_num, ..., broadcast_num, seq} or {bs, seq}
  pad_dst_shape_ = {bs, seq};
  // pad to 3 dimensions because that is what the runtime code requires
  int64_t runtime_dims = 3;
  int64_t broadcast_nums = 1;
  if (dst_shape.size() >= runtime_dims) {
    broadcast_nums =
        std::accumulate(dst_shape.begin() + 1, dst_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
  }
  pad_dst_shape_.insert(pad_dst_shape_.begin() + 1, broadcast_nums);
  pad_dst_stride_ = GetStrides(pad_dst_shape_);

  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);
}

void PaddingSequenceOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& mask_data = static_cast<const int32_t*>(input[0]->data());
  // when change data value please use mutable_data
  auto dst_data = static_cast<float*>(output[0]->mutable_data());
  LOG_IF(ERROR, reinterpret_cast<void*>(dst_data) == reinterpret_cast<void*>(const_cast<int32_t*>(mask_data)))
      << "DST ptr should not be equal to SRC ptr.";

  // Prepare actual sequence length whose element is 1
  std::vector<int> seqs_per_batch(src_shape_[0]);
#pragma omp parallel for
  for (int i = 0; i < src_shape_[0]; ++i) {
    for (int j = src_shape_[1] - 1; j >= 0; --j) {
      auto idx = i * src_stride_[0] + j;
      if (mask_data[idx] == 0) {
        continue;
      }
      seqs_per_batch[i] = j + 1;
      break;
    }
  }

  // 1. Execute the dst
  for (int i = 0; i < pad_dst_shape_[0]; ++i) {
#pragma omp parallel for
    for (int j = 0; j < pad_dst_shape_[1]; ++j) {
      int row_idx = i * pad_dst_stride_[0] + j * pad_dst_stride_[1];
      memset(dst_data + row_idx, 0, sizeof(float) * seqs_per_batch[i]);
#pragma omp simd
      for (int k = seqs_per_batch[i]; k < pad_dst_shape_[2]; ++k) {
        dst_data[row_idx + k] = padding_;
      }
    }
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(PaddingSequence);
}  // namespace executor
