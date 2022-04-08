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

#include "matmul.hpp"

#include "operator_registry.hpp"
namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"int32", dnnl::memory::data_type::s32},
    {"s32", dnnl::memory::data_type::s32},  {"fp16", dnnl::memory::data_type::f16},
    {"u8", dnnl::memory::data_type::u8},    {"s8", dnnl::memory::data_type::s8},
    {"bf16", dnnl::memory::data_type::bf16}};

MatmulOperator::MatmulOperator(const OperatorConfig& conf)
    : Operator(conf),
      src0_perm_({}),
      src1_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      format_any_(true),
      cache_weight_(false) {
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
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("cache_weight");
  if (iter != attrs_map.end()) {
    cache_weight_ = attrs_map["cache_weight"] == "true";
  }
  iter = attrs_map.find("append_op");
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  append_eltwise_ = gelu_erf_ || gelu_tanh_ || tanh_;
}

MatmulOperator::~MatmulOperator() {}

void MatmulOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
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
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 4: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      has_bias_ = true;
      break;
    }
    case 6: {
      src0_ = input[0];
      src1_ = input[1];
      src0_min_ = input[2];
      src0_max_ = input[3];
      src1_min_ = input[4];
      src1_max_ = input[5];
      break;
    }
    case 7: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src0_min_ = input[3];
      src0_max_ = input[4];
      src1_min_ = input[5];
      src1_max_ = input[6];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 8: {
      if (append_sum_ || binary_add_) {
        // dynamic quantization
        src0_ = input[0];
        src1_ = input[1];
        bias_ = input[2];
        post_ = input[3];
        src0_min_ = input[4];
        src0_max_ = input[5];
        src1_min_ = input[6];
        src1_max_ = input[7];
        has_bias_ = true;
      } else {
        // static quantization
        src0_ = input[0];
        src1_ = input[1];
        src0_min_ = input[2];
        src0_max_ = input[3];
        src1_min_ = input[4];
        src1_max_ = input[5];
        dst_min_ = input[6];
        dst_max_ = input[7];
        has_bias_ = false;
      }
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
      has_bias_ = !(append_sum_ || binary_add_);
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
      has_bias_ = true;
      break;
    }
  }
}

void MatmulOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
  is_dynamic_ = output.size() > 1 || (src0_min_ != nullptr && src0_min_->raw_data() == nullptr);
  if (is_dynamic_) LOG(INFO) << this->name() << " is DYNAMIC!!!";
  if (!is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    int ic_dim = 0;
    vector<float> rescales;
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr)
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      rescales = GetRescales(src0_scales, src1_scales, dst_scales_, dst_->dtype(), append_eltwise_);
    } else {
      rescales = vector<float>(1, 1.f);
    }
    if (output_scale_ != 1.f) {
      for (int i = 0; i < rescales.size(); i++) {
        rescales[i] *= output_scale_;
      }
    }
    attr_.set_output_scales(ic_dim, rescales);
  }
}

// 1. Create primitive
void MatmulOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src1_shape_origin = src1_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_shape = GetShapes(src1_shape_origin, src1_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin, src1_perm_);
  src0_->set_shape(src0_shape);
  src1_->set_shape(src1_shape);

  memory::dims bias_shape;
  if (has_bias_) bias_shape = bias_->shape();

  // 1.2 malloc tensor for output
  // src0_: M*K, src1_: K*N, DST: M*N
  vector<int64_t> dst_shape_origin = src0_shape;
  dst_shape_origin.back() = src1_shape.back();

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight forward
  // post transpose will use origin shape and that means the dst buffer in matmul
  // is a buffer transposed back from dst_perm(understand tranpose to and transpose back)
  // pre_transpose: src_buffer -> pre_transpose -> target_buffer in matmul
  // post_transpose: target_buffer in matmul<- post transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);
  if (has_bias_ && bias_shape.size() != dst_shape.size()) {
    bias_shape = vector<int64_t>(dst_shape.size(), 1);
    bias_shape.back() = bias_->shape().back();
  }
  vector<int64_t> bias_stride = GetStrides(bias_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  memory::desc src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], src1_stride);

  memory::desc any_dst_md, dst_md;
  if (is_dynamic_) {
    // matmul output dtype in dynamic quantization should be fp32 and then manually quantize to u8/s8.
    any_dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], dst_stride);
  } else {
    any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
  }

  memory::desc bias_md;
  memory::desc any_bias_md;
  if (has_bias_) {
    bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
  }

  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }

  if (cache_weight_) {
    src1_md = dnnl::memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  }

  // 2.2 Prepare op descriptors
  dnnl::matmul::desc matmul_d =
      has_bias_ ? dnnl::matmul::desc(src0_md, src1_md, bias_md, dst_md) : dnnl::matmul::desc(src0_md, src1_md, dst_md);

  if (format_any_) {
    matmul_d = has_bias_ ? dnnl::matmul::desc(any_src0_md, any_src1_md, any_bias_md, any_dst_md)
                         : dnnl::matmul::desc(any_src0_md, any_src1_md, any_dst_md);
  }

  // 2.3 Prepare primitive descriptors (cached)
  vector<float> src0_scales;
  vector<float> src1_scales;
  vector<float> dst_scales;
  vector<float> rescales;
  dnnl::post_ops po;
  int ic_dim = 0;
  if (is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 2 : 0;
      // attr.set_output_scales(ic_dim, {DNNL_RUNTIME_F32_VAL});
      vector<int64_t> scale_shape(src1_shape.size(), 1);
      scale_shape[src1_shape.size() - 1] = src1_min_->size();
      scale_md_ = memory::desc(scale_shape, memory::data_type::f32, GetStrides(scale_shape));
      po.append_binary(algorithm::binary_mul, scale_md_);
      // need zero point when src0 is u8
      if (src0_->dtype() == "u8") {
        attr_.set_zero_points(DNNL_ARG_SRC, ic_dim, {DNNL_RUNTIME_S32_VAL});
      }
    }
  }
  if (append_sum_) {
    float beta = 1.0;
    po.append_sum(beta);
  }
  if (gelu_erf_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_erf, op_alpha, op_beta);
  }
  if (gelu_tanh_) {
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
  if (!is_dynamic_ && dst_->dtype() == "u8" && dst_min_->data() != nullptr) {
    if (append_eltwise_) {
      float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
      po.append_eltwise(dst_scales_[0], algorithm::eltwise_linear, 1., zero_point);
    } else {
      vector<int> dst_zero_points;
      dst_zero_points = GetZeroPoints(dst_min_->data(), dst_scales_, dst_->dtype());
      attr_.set_zero_points(DNNL_ARG_DST, ic_dim, dst_zero_points);
    }
  }
  if (binary_add_) {
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    binary_m_ = memory(binary_md, eng_);
  }
  attr_.set_post_ops(po);
  matmul_pd_ = dnnl::matmul::primitive_desc(matmul_d, attr_, eng_);

  // 2.4 Prepare primitive objects (cached)
  matmul_p_ = dnnl::matmul(matmul_pd_);

  // 2.5 Prepare memory objects (cached)
  src0_m_ = memory(src0_md, eng_);
  dst_m_ = memory(dst_md, eng_);
  if (has_bias_) {
    bias_m_ = memory(bias_md, eng_, const_cast<void*>(bias_->data()));
    memory any_bias_m = bias_m_;
    if (matmul_pd_.bias_desc() != bias_m_.get_desc()) {
      any_bias_m = memory(matmul_pd_.bias_desc(), eng_);
      dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
    }
    memory_args_[DNNL_ARG_BIAS] = any_bias_m;
  }
  if (cache_weight_) {
    memory::desc user_src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::ab);
    src1_m_ = memory(user_src1_md, eng_, const_cast<void*>(src1_->data()));
    memory any_src1_m = src1_m_;
    if (matmul_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m = memory(matmul_pd_.weights_desc(), eng_);
      dnnl::reorder(src1_m_, any_src1_m).execute(eng_stream_, src1_m_, any_src1_m);
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
  } else {
    src1_m_ = memory(src1_md, eng_);
  }
}

// 2. inference kernel(for int8 and f32)
void MatmulOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src0_data = src0_->data();
  const auto& src1_data = src1_->data();
  void* dst_data;
  // create a dynamic quantization output with fp32.
  Tensor matmul_fp32_res;
  if (is_dynamic_) {
    matmul_fp32_res = *dst_;
    matmul_fp32_res.set_dtype("fp32");
    dst_data = matmul_fp32_res.mutable_data();
  } else {
    dst_data = dst_->mutable_data();
  }
  // has post_op: append_sum
  if (post_ != nullptr && !binary_add_) {
    LOG(INFO) << "matmul has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1) {
      post_->unref_data(true);
      if (is_dynamic_)
        matmul_fp32_res.set_data(post_data_ptr);
      else
        dst_->set_data(post_data_ptr);
      dst_data = post_data_ptr;
    } else {
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }

  // 1. Prepare memory objects with data_ptr
  src0_m_.set_data_handle(const_cast<void*>(src0_data), eng_stream_);
  src1_m_.set_data_handle(const_cast<void*>(src1_data), eng_stream_);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), eng_stream_);

  memory any_src0_m = src0_m_;
  memory any_src1_m = src1_m_;
  memory any_dst_m = dst_m_;

  // 2. Reorder the data when the primitive memory and user memory are different
  if (matmul_pd_.src_desc() != src0_m_.get_desc()) {
    any_src0_m = memory(matmul_pd_.src_desc(), eng_);
    dnnl::reorder(src0_m_, any_src0_m).execute(eng_stream_, src0_m_, any_src0_m);
  }
  if (!cache_weight_) {
    if (matmul_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m = memory(matmul_pd_.weights_desc(), eng_);
      dnnl::reorder(src1_m_, any_src1_m).execute(eng_stream_, src1_m_, any_src1_m);
    }
  }
  if (matmul_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(matmul_pd_.dst_desc(), eng_);
  }

  // the runtime calculation of dynamic quantization
  vector<int32_t> src0_zero_points;
  vector<float> rescales;
  vector<float> dynamic_bias;
  if (is_dynamic_) DynamicForward(&src0_zero_points, &rescales, &dynamic_bias);

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  if (!cache_weight_) memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;

  // has post_op: binary_add
  if (post_ != nullptr && binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(reinterpret_cast<void*>(post_ptr), eng_stream_);
    // dynamic quantization inserts additional post_ops
    int op_idx = 0;
    if (is_dynamic_) {
      op_idx++;
    }
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(op_idx) | DNNL_ARG_SRC_1] = binary_m_;
  }

  // 4. Execute the primitive
  matmul_p_.execute(eng_stream_, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  if (matmul_pd_.dst_desc() != dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, dst_m_).execute(eng_stream_, any_dst_m, dst_m_);
  }
  eng_stream_.wait();
  // 6. unref tensors
  this->unref_tensors(input);

  if (is_dynamic_) {
    // quantize the fp32 result of matmul
    if (output.size() > 1) {
      RuntimeMinmax();
      // quantize
      if (output_dtype_ == "u8" || output_dtype_ == "s8") {
        auto scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
#if __AVX512F__
        Quantize_avx512(matmul_fp32_res.size(), dst_->dtype(), matmul_fp32_res.data(),
                        static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#else
        Quantize(matmul_fp32_res.size(), dst_->dtype(), matmul_fp32_res.data(),
                 static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#endif
        matmul_fp32_res.unref_data();
      } else {
        // copy fp32_res to dst if not quantize
        void* res_ptr = const_cast<void*>(matmul_fp32_res.data());
        matmul_fp32_res.unref_data(true);
        dst_->set_data(res_ptr);
      }
    } else {
      void* res_ptr = const_cast<void*>(matmul_fp32_res.data());
      matmul_fp32_res.unref_data(true);
      dst_->set_data(res_ptr);
    }
  }
}

void MatmulOperator::RuntimeMinmax() {
  // use onednn reduction calculate min/max
  vector<int64_t> reduce_shape(dst_->shape().size(), 1);
  vector<int64_t> reduce_stride = GetStrides(reduce_shape);
  memory::desc dst_md(reduce_shape, memory::data_type::f32, reduce_stride);
  memory reduce_min(dst_md, eng_);
  memory reduce_max(dst_md, eng_);
  reduce_min.set_data_handle(dst_min_->mutable_data());
  reduce_max.set_data_handle(dst_max_->mutable_data());
  dnnl::reduction::desc reduce_min_d(algorithm::reduction_min, dst_m_.get_desc(), dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_min_pd(reduce_min_d, eng_);
  dnnl::reduction(reduce_min_pd).execute(eng_stream_, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_min}});
  dnnl::reduction::desc reduce_max_d(algorithm::reduction_max, dst_m_.get_desc(), dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_max_pd(reduce_max_d, eng_);
  dnnl::reduction(reduce_max_pd).execute(eng_stream_, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_max}});
}

void MatmulOperator::DynamicForward(vector<int32_t>* src0_zero_points_ptr, vector<float>* rescales_ptr,
                                    vector<float>* dynamic_bias_ptr) {
  auto& src0_zero_points = *src0_zero_points_ptr;
  auto& rescales = *rescales_ptr;
  auto& dynamic_bias = *dynamic_bias_ptr;
  memory scale_f32_mem(scale_md_, eng_);
  memory zp_src0_mem({{1}, memory::data_type::s32, {1}}, eng_);
  int channel_size = src1_min_->size();  // channel_size=1 represent per_tensor
  rescales.resize(channel_size);
  vector<float> src0_scales;
  vector<float> src1_scales;
  src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
  src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
  if (channel_size == 1) {
    rescales[0] = output_scale_ / src0_scales[0] / src1_scales[0];
  } else {
#pragma omp parallel for
    for (int i = 0; i < channel_size; i++) rescales[i] = output_scale_ / src0_scales[0] / src1_scales[i];
  }
  scale_f32_mem.set_data_handle(reinterpret_cast<void*>(rescales.data()), eng_stream_);
  memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = scale_f32_mem;
  // memory_args_[DNNL_ARG_ATTR_OUTPUT_SCALES] = scale_f32_mem;

  // The bias loaded from file is not scaled. So need rescaled runtime.
  if (has_bias_) {
    dynamic_bias.resize(bias_->size());
    float* bias_data = reinterpret_cast<float*>(bias_m_.get_data_handle());
    if (channel_size == 1) {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[0];
    } else {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[i];
    }
    bias_m_.set_data_handle(reinterpret_cast<void*>(dynamic_bias.data()), eng_stream_);
  }

  if (src0_->dtype() == "u8") {
    src0_zero_points = GetZeroPoints(src0_min_->data(), src0_scales, src0_->dtype());
    zp_src0_mem.set_data_handle(reinterpret_cast<void*>(src0_zero_points.data()), eng_stream_);
    memory_args_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = zp_src0_mem;
  }
}
REGISTER_OPERATOR_CLASS(Matmul);
}  // namespace executor
