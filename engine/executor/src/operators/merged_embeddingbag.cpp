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
#include "merged_embeddingbag.hpp"

namespace executor {

MergedEmbeddingbagOperator::MergedEmbeddingbagOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("mode");
  mode_ = (iter != attrs_map.end()) ? iter->second : "";
}

void MergedEmbeddingbagOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  assert(input.size() == output.size() + 2);
  for (int i = 0; i < output.size(); i++) {
    const string weight_dtype = input[i + 2]->dtype();
    output[i]->set_dtype(weight_dtype);
  }
}

void MergedEmbeddingbagOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> offset_shape = input[1]->shape();
  for (int i = 0; i < output.size(); i++) {
    const vector<int64_t> weight_shape = input[i + 2]->shape();
    vector<int64_t> dst_shape = {offset_shape[1], weight_shape[1]};
    output[i]->set_shape(dst_shape);
  }
}

void MergedEmbeddingbagOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* offsets = input[0];
  Tensor* indices = input[1];
  vector<Tensor*> weights;
  weights.assign(input.begin() + 2, input.end());

  int64_t n_tables = weights.size();
  int64_t bs = offsets->shape()[1];
  if (n_tables != offsets->shape()[0] || n_tables != indices->shape()[0]) {
    LOG(ERROR) << "weights size: " << n_tables << ", offset shape 0: " << offsets->shape()[0]
               << ", indices shape 0: " << indices->shape()[0];
  }

  vector<void*> weights_ptr;
  vector<string> dtypes;
  for (auto& w : weights) {
    weights_ptr.emplace_back(w->mutable_data());
    dtypes.emplace_back(w->dtype());
  }
  vector<void*> outs_ptr;
  for (auto& o : output) {
    outs_ptr.emplace_back(o->mutable_data());
  }

  int32_t* offsets_data = reinterpret_cast<int32_t*>(offsets->mutable_data());
  int32_t* indices_data = reinterpret_cast<int32_t*>(indices->mutable_data());

#pragma omp parallel for
  for (int table_id = 0; table_id < n_tables; ++table_id) {
#pragma omp simd
    for (int n = 0; n < bs; ++n) {
      int32_t offset_idx = n + table_id * bs;
      int32_t pool_begin = offsets_data[offset_idx] + table_id * bs;
      int32_t pool_end = ((offset_idx + 1) % bs == 0) ? (table_id + 1) * indices->shape()[1]
                                                      : offsets_data[offset_idx + 1] + table_id * bs;
      int64_t feature_size = weights[table_id]->shape()[1];
      if (dtypes[table_id] == "fp32") {
        float* out_ptr = &((reinterpret_cast<float*>(outs_ptr[table_id]))[n * feature_size]);
        float* weight_ptr = reinterpret_cast<float*>(weights_ptr[table_id]);
        emb_pooling_ker<float>(out_ptr, weight_ptr, pool_begin, pool_end, feature_size, indices_data, mode_);
      } else if (dtypes[table_id] == "bf16") {
        uint16_t* out_ptr = &((reinterpret_cast<uint16_t*>(outs_ptr[table_id]))[n * feature_size]);
        uint16_t* weight_ptr = reinterpret_cast<uint16_t*>(weights_ptr[table_id]);
        emb_pooling_ker<uint16_t>(out_ptr, weight_ptr, pool_begin, pool_end, feature_size, indices_data, mode_);
      } else if (dtypes[table_id] == "u8") {
        uint8_t* out_ptr = &((reinterpret_cast<uint8_t*>(outs_ptr[table_id]))[n * feature_size]);
        uint8_t* weight_ptr = reinterpret_cast<uint8_t*>(weights_ptr[table_id]);
        emb_pooling_ker<uint8_t>(out_ptr, weight_ptr, pool_begin, pool_end, feature_size, indices_data, mode_);
      } else {
        LOG(ERROR) << "Merged embedding can not support dtype: " << dtypes[table_id];
      }
    }
  }

  this->unref_tensors(input);
}

template <typename T>
void emb_pooling_ker(T* out, T* in, const size_t pool_begin, const size_t pool_end, const size_t vector_size,
                     int32_t* indices_data, const string& mode) {
  auto idx = indices_data[pool_begin];
  auto weight_ptr = &in[idx * vector_size];
  if (pool_end - pool_begin == 1) {
    move_ker(out, weight_ptr, vector_size);
  } else {
    // add if there is more than 1 indice in this bag, need accumulate to float
    // buffer
    T* temp_out = reinterpret_cast<T*>(malloc(vector_size * sizeof(T)));
    if (temp_out != nullptr) {
      zero_ker(temp_out, vector_size);
      for (auto p = pool_begin; p < pool_end; ++p) {
        idx = indices_data[p];
        weight_ptr = &in[idx * vector_size];
        add_ker(temp_out, weight_ptr, vector_size);
      }
      if (mode == "mean") {
        auto L = pool_end - pool_begin;
        const uint8_t scale_factor = 1.0 / L;
#pragma omp simd
        for (int d = 0; d < vector_size; ++d) {
          temp_out[d] = scale_factor * temp_out[d];
        }
      }
      move_ker(out, temp_out, vector_size);
    }
    free(temp_out);
  }
}
template void emb_pooling_ker(float* out, float* in, const size_t pool_begin, const size_t pool_end,
                              const size_t vector_size, int32_t* indices_data, const string& mode);
template void emb_pooling_ker(uint16_t* out, uint16_t* in, const size_t pool_begin, const size_t pool_end,
                              const size_t vector_size, int32_t* indices_data, const string& mode);
template void emb_pooling_ker(uint8_t* out, uint8_t* in, const size_t pool_begin, const size_t pool_end,
                              const size_t vector_size, int32_t* indices_data, const string& mode);

REGISTER_OPERATOR_CLASS(MergedEmbeddingbag);
}  // namespace executor
