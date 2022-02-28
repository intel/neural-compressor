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

#include "inner_product.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

InnerProductOperator::InnerProductOperator(const OperatorConfig& conf)
    : Operator(conf),
      src0_perm_({}),
      src1_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      format_any_(true),
      gelu_split_(false),
      weight_cached_(false),
      has_bias_(false) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("src0_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src0_perm_, attrs_map["src0_perm"], ",");
  }
  iter = attrs_map.find("src1_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src1_perm_, attrs_map["src1_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("output_scale");
  if (iter != attrs_map.end()) {
    output_scale_ = StringToNum<float>(attrs_map["output_scale"]);
  }
  iter = attrs_map.find("format_any");
  if (iter != attrs_map.end()) {
    format_any_ = attrs_map["format_any"] == "true";
  }
  iter = attrs_map.find("is_symm");
  if (iter != attrs_map.end()) {
    is_asymm_ = attrs_map["is_symm"] == "true";
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("gelu_split");
  if (iter != attrs_map.end()) {
    gelu_split_ = attrs_map["gelu_split"] == "true";
  }
  iter = attrs_map.find("append_op");
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  sigmoid_ = (iter != attrs_map.end() && iter->second == "sigmoid") ? true : false;
  relu_ = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  append_eltwise_ = (gelu_erf_ && !gelu_split_) || (gelu_tanh_ && !gelu_split_) || tanh_ || sigmoid_ || relu_;
  append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  LOG(INFO) << "append_op: " << append_op_;
}

InnerProductOperator::~InnerProductOperator() {}

void InnerProductOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  switch (input_size) {
    case 2: {
      src0_ = input[0];
      src1_ = input[1];
      break;
    }
    case 3: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      break;
    }
    case 4: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      break;
    }
    case 8: {
      src0_ = input[0];
      src1_ = input[1];
      src0_min_ = input[2];
      src0_max_ = input[3];
      src1_min_ = input[4];
      src1_max_ = input[5];
      dst_min_ = input[6];
      dst_max_ = input[7];
      break;
    }
    case 9: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src0_min_ = input[3];
      src0_max_ = input[4];
      src1_min_ = input[5];
      src1_max_ = input[6];
      dst_min_ = input[7];
      dst_max_ = input[8];
      break;
    }
    case 10: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      src0_min_ = input[4];
      src0_max_ = input[5];
      src1_min_ = input[6];
      src1_max_ = input[7];
      dst_min_ = input[8];
      dst_max_ = input[9];
      break;
    }
  }
}

void InnerProductOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  has_bias_ = (input.size() == 4 || input.size() == 10) ||
                      ((input.size() == 3 || input.size() == 9) && !append_sum_ && !binary_add_)
                  ? true
                  : false;
  LOG(INFO) << "inner product has bias add " << has_bias_;
  dst_->set_dtype(output_dtype_);
  if (src1_->dtype() == "fp32") {
    weight_zero_ratio_ = GetSparseRatio<float>(static_cast<const float*>(src1_->data()), src1_->shape(), blocksize_);
  } else if (src1_->dtype() == "s8") {
    blocksize_ = {4, 16};
    weight_zero_ratio_ = GetSparseRatio<int8_t>(static_cast<const int8_t*>(src1_->data()), src1_->shape(), blocksize_);
  } else if (src1_->dtype() != "bf16") {
    LOG(ERROR) << "src1 in innerproduct can not support dtype: " << src1_->dtype();
  }
  LOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  int64_t N = src1_->shape()[1];
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
    N = src1_->shape()[0];
  }
  if (weight_zero_ratio_ < sparse_threshold_ || N % blocksize_[1] != 0) {
    dense_flag_ = true;
  }
#ifndef __AVX512F__
  if (!dense_flag_) {
    if (!src1_) {
      dense_flag_ = true;
      LOG(ERROR) << "Sparse fp32 kernel in InnerProduct only supports AVX512!";
    } else {
#ifndef __AVX512VNNI__
      dense_flag_ = true;
      LOG(ERROR) << "Sparse int8 kernel in InnerProduct only supports AVX512VNNI!";
#endif
    }
  }
#endif
  if (dense_flag_) {
    PrepareDense(input, output);
  } else {
    PrepareSparse(input, output);
  }
}

void InnerProductOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (dense_flag_) {
    ReshapeDense(input, output);
  } else {
    ReshapeSparse(input, output);
  }
}

void InnerProductOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (dense_flag_) {
    ForwardDense(input, output);
  } else {
#if __AVX512F__
    ForwardSparse(input, output);
#endif
  }
  this->unref_tensors(input);
}

void InnerProductOperator::PrepareSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> weight_shape = src1_->shape();
  vector<int64_t> weight_shape_perm = weight_shape;
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
    weight_shape_perm = {weight_shape[1], weight_shape[0]};
  }
  src1_->set_shape(weight_shape_perm);
  if (!src1_min_) {  // fp32 kernel prepare
    const float* weight_data = static_cast<const float*>(src1_->data());
    float* weight_data_perm = static_cast<float*>(malloc(src1_->size() * sizeof(float)));
    memcpy(weight_data_perm, weight_data, src1_->size() * sizeof(float));
    if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
      TransposeMatrix<float>(weight_data, weight_shape, weight_data_perm);
    }
    sparse_weight_ = create_bsc_matrix<float>(weight_data_perm, weight_shape_perm, blocksize_);
    free(weight_data_perm);
  } else {  // int8 kernel prepare
    const int8_t* weight_data = static_cast<const int8_t*>(src1_->data());
    int8_t* weight_data_perm = static_cast<int8_t*>(malloc(src1_->size() * sizeof(int8_t)));
    memcpy(weight_data_perm, weight_data, src1_->size() * sizeof(int8_t));
    if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
      TransposeMatrix<int8_t>(weight_data, weight_shape, weight_data_perm);
    }
    sparse_weight_int8_ = create_bsc_matrix<int8_t>(weight_data_perm, weight_shape_perm, blocksize_);
    reorder_bsc_int8_4x16(sparse_weight_int8_);
    free(weight_data_perm);
    vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
    vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
    vector<float> dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
    vector<float> rescales;
    for (int i = 0; i < src1_scales.size(); i++) {
      rescales.emplace_back(dst_scales[0] / (src0_scales[0] * src1_scales[i]));
    }
    rescales_ = rescales;
  }
}

void InnerProductOperator::ReshapeSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = src0_shape_origin;
  if (!src0_perm_.empty() && src0_perm_ == vector<int64_t>{0, 1}) {
    src0_shape = {src0_shape[1], src0_shape[0]};
  }
  src0_->set_shape(src0_shape);

  vector<int64_t> dst_shape = {src0_shape[0], src1_->shape()[1]};
  dst_->set_shape(dst_shape);
}

#if __AVX512F__
void InnerProductOperator::ForwardSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int64_t M = src0_->shape()[0];
  int64_t N = src1_->shape()[1];
  int64_t K = src0_->shape()[1];
  // fp32 kernel
  if (!src1_min_) {
    const int64_t* rowidxs = sparse_weight_->rowidxs;
    const int64_t* colptr = sparse_weight_->colptr;
    const int64_t ncolptr = sparse_weight_->ncolptr;
    const float* A = static_cast<const float*>(src0_->data());
    const float* B = static_cast<const float*>(sparse_weight_->data);
    float* C = static_cast<float*>(dst_->mutable_data());
    if (has_bias_) {
      const float* bias = static_cast<const float*>(bias_->data());
      if (append_op_ == "") {
        sparse_gemm_bsc_bias_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "relu") {
        sparse_gemm_bsc_bias_relu_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "sum") {
        const float* post = static_cast<const float*>(post_->data());
        sparse_gemm_bsc_bias_sum_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, post, C, M_NBLK_);
      } else if (append_op_ == "tanh") {
        sparse_gemm_bsc_bias_tanh_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "gelu_tanh") {
        sparse_gemm_bsc_bias_gelu_tanh_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "sigmoid") {
        sparse_gemm_bsc_bias_sigmod_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else {
        LOG(INFO) << "inner product has no such sparse kernel, output tensor is" << output[0]->name();
      }
    } else {
      if (append_op_ == "") {
        sparse_gemm_bsc_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, C, M_NBLK_);
      }
    }
  } else {  // int8 kernel
#if __AVX512VNNI__
    const int64_t* rowidxs = sparse_weight_int8_->rowidxs;
    const int64_t* colptr = sparse_weight_int8_->colptr;
    const int64_t ncolptr = sparse_weight_int8_->ncolptr;
    const uint8_t* A = static_cast<const uint8_t*>(src0_->data());
    const int8_t* B = static_cast<const int8_t*>(sparse_weight_int8_->data);
    if (src1_->size() > 1) {  // per channel kernel
      if (output[0]->dtype() == "u8") {
        uint8_t* C = static_cast<uint8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8u8_pc_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_, C,
                                                M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "s8") {
        int8_t* C = static_cast<int8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8s8_pc(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_, C,
                                           M_NBLK_);
          }
        }
      }
    } else {  // per tensor kernel
      if (output[0]->dtype() == "fp32") {
        float* C = static_cast<float*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                         M_NBLK_);
          } else if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8f32_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0],
                                              C, M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "u8") {
        uint8_t* C = static_cast<uint8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8u8_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                             M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "s8") {
        int8_t* C = static_cast<int8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8s8(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                        M_NBLK_);
          }
        }
      }
    }
#endif
  }
}
#endif

void InnerProductOperator::PrepareDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dnnl::primitive_attr attr;
  vector<float> src0_scales;
  vector<float> src1_scales;
  vector<float> dst_scales;
  vector<float> rescales;
  if (output_scale_ != 1.f) {
    attr.set_output_scales(0, {output_scale_});
  } else if (src1_min_ != nullptr) {
    const int ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
    src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
    src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
    dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
    rescales = GetRescales(src0_scales, src1_scales, dst_scales, dst_->dtype(), append_eltwise_);
    attr.set_output_scales(ic_dim, rescales);
  }

  dnnl::post_ops po;
  if (append_sum_) {
    float beta = 1.0;
    po.append_sum(beta);
  }

  if (gelu_erf_ && !gelu_split_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_erf, op_alpha, op_beta);
  }
  if (gelu_tanh_ && !gelu_split_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_tanh, op_alpha, op_beta);
  }
  if (tanh_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_tanh, op_alpha, op_beta);
  }
  if (sigmoid_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_logistic, op_alpha, op_beta);
  }
  if (relu_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_relu, op_alpha, op_beta);
  }
  // this is to sub zero point in fp32 to make the output u8/s8
  if (append_eltwise_ && (dst_->dtype() == "u8" || dst_->dtype() == "s8")) {
    float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
    po.append_eltwise(dst_scales[0], algorithm::eltwise_linear, 1., zero_point);
  }
  if (append_eltwise_ || append_sum_) attr.set_post_ops(po);
  attr_ = attr;

  // cache weight here, save weight and bias memory descriptor
  vector<int64_t> src1_shape_origin = src1_->shape();
  vector<int64_t> src1_shape = GetShapes(src1_shape_origin, src1_perm_);
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin, src1_perm_);
  src1_->set_shape(src1_shape);
  any_src1_md_ = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  src1_md_ = memory::desc(src1_shape, type2mem[src1_->dtype()], src1_stride);
  src1_m_ = memory(src1_md_, eng_, src1_->mutable_data());

  if (has_bias_) {
    vector<int64_t> bias_shape = {src1_shape[0]};
    vector<int64_t> bias_stride = GetStrides(bias_shape);
    bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
    bias_m_ = memory(bias_md_, eng_, bias_->mutable_data());
  }
}

// 1. Create primitive
void InnerProductOperator::ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  src0_->set_shape(src0_shape);

  // 1.2 malloc tensor for output
  // src0_: M*K, src1_: K*N, DST: M*N
  vector<int64_t> dst_shape_origin = {src0_shape[0], src1_->shape()[0]};

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight forward
  // post transpose will use origin shape and that means the dst buffer in matmul
  // is a buffer transposed back from dst_perm(understand tranpose to and transpose back)
  // pre_transpose: src0_buffer -> pre_transpose -> target_buffer in matmul
  // post_transpose: target_buffer in matmul<- post transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
  memory::desc dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);

  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);

  // 2.2 Prepare op descriptors
  dnnl::inner_product_forward::desc inner_product_d =
      has_bias_ ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, bias_md_, dst_md)
                : dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, dst_md);

  if (format_any_) {
    inner_product_d = has_bias_ ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                                    any_src1_md_, any_bias_md_, any_dst_md)
                                : dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                                    any_src1_md_, any_dst_md);
  }

  if (gelu_erf_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_erf, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_);
  }
  if (gelu_tanh_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_tanh, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_);
  }
  if (binary_add_) {
    dnnl::primitive_attr attr;
    dnnl::post_ops po;
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    attr.set_post_ops(po);
    binary_m_ = memory(binary_md, eng_);
    attr_ = attr;
  }

  inner_product_pd_ = dnnl::inner_product_forward::primitive_desc(inner_product_d, attr_, eng_);

  // 2.4 Prepare primitive objects (cached)
  inner_product_p_ = dnnl::inner_product_forward(inner_product_pd_);

  // 2.5 Prepare memory objects (cached)
  src0_m_ = memory(src0_md, eng_);
  dst_m_ = memory(dst_md, eng_);
  if (!weight_cached_) {
    memory any_src1_m = src1_m_;
    if (inner_product_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m = memory(inner_product_pd_.weights_desc(), eng_);
      dnnl::reorder(src1_m_, any_src1_m).execute(eng_stream_, src1_m_, any_src1_m);
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
    if (has_bias_) {
      memory any_bias_m = bias_m_;
      if (inner_product_pd_.bias_desc() != bias_m_.get_desc()) {
        any_bias_m = memory(inner_product_pd_.bias_desc(), eng_);
        dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
      }
      memory_args_[DNNL_ARG_BIAS] = any_bias_m;
    }
    weight_cached_ = true;
  }
}

// 2. inference kernel(for int8 and f32)
void InnerProductOperator::ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (post_ != nullptr && !binary_add_) {
    LOG(INFO) << "inner product has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1) {
      post_->unref_data(true);
      dst_->set_data(post_data_ptr);
    } else {
      void* dst_data_ptr = dst_->mutable_data();
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data_ptr, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }

  // 0. Alias variables part
  const auto& src0_data = src0_->data();
  // when change data value please use mutable_data
  auto dst_data = dst_->mutable_data();

  // 1. Prepare memory objects with data_ptr
  src0_m_.set_data_handle(const_cast<void*>(src0_data), eng_stream_);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), eng_stream_);

  memory any_src0_m = src0_m_;
  memory any_dst_m = dst_m_;

  // 2. Reorder the data when the primitive memory and user memory are different
  if (inner_product_pd_.src_desc() != src0_m_.get_desc()) {
    any_src0_m = memory(inner_product_pd_.src_desc(), eng_);
    dnnl::reorder(src0_m_, any_src0_m).execute(eng_stream_, src0_m_, any_src0_m);
  }

  if (inner_product_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(inner_product_pd_.dst_desc(), eng_);
  }

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;
  if (binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(post_ptr, eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_m_;
  }

  // 4. Execute the primitive
  inner_product_p_.execute(eng_stream_, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  if (inner_product_pd_.dst_desc() != dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, dst_m_).execute(eng_stream_, any_dst_m, dst_m_);
  }

  // gelu seperate
  if ((gelu_split_ && gelu_tanh_) || (gelu_split_ && gelu_erf_)) {
    dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), gelu_eng_stream_);
    gelu_m_.set_data_handle(reinterpret_cast<void*>(dst_data), gelu_eng_stream_);
    gelu_memory_args_[DNNL_ARG_SRC] = dst_m_;
    gelu_memory_args_[DNNL_ARG_DST] = gelu_m_;
    gelu_p_.execute(gelu_eng_stream_, gelu_memory_args_);
  }
  eng_stream_.wait();
}

REGISTER_OPERATOR_CLASS(InnerProduct);
}  // namespace executor
