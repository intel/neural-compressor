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
  iter = attrs_map.find("is_symm");
  if (iter != attrs_map.end()) {
    is_asymm_ = attrs_map["is_symm"] == "true";
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

void MatmulOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
}

// 1. Create primitive
void MatmulOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  bool has_bias =
      (input.size() == 4 || input.size() == 10) ||
              ((input.size() == 3 || input.size() == 7 || input.size() == 9) && !append_sum_ && !binary_add_)
          ? true
          : false;

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
  if (has_bias) bias_shape = bias_->shape();

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
  if (has_bias && bias_shape.size() != dst_shape.size()) {
    bias_shape = vector<int64_t>(dst_shape.size(), 1);
    bias_shape.back() = bias_->shape().back();
  }
  vector<int64_t> bias_stride = GetStrides(bias_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  memory::desc src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], src1_stride);

  memory::desc any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
  memory::desc dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);

  memory::desc bias_md;
  memory::desc any_bias_md;
  if (has_bias) {
    bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
  }

  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);
  if (cache_weight_) {
    src1_md = dnnl::memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  }

  // 2.2 Prepare op descriptors
  dnnl::matmul::desc matmul_d =
      has_bias ? dnnl::matmul::desc(src0_md, src1_md, bias_md, dst_md) : dnnl::matmul::desc(src0_md, src1_md, dst_md);

  if (format_any_) {
    matmul_d = has_bias ? dnnl::matmul::desc(any_src0_md, any_src1_md, any_bias_md, any_dst_md)
                        : dnnl::matmul::desc(any_src0_md, any_src1_md, any_dst_md);
  }

  // 2.3 Prepare primitive descriptors (cached)
  dnnl::primitive_attr attr;
  vector<float> src0_scales;
  vector<float> src1_scales;
  vector<float> dst_scales;
  vector<float> rescales;
  int ic_dim = 0;
  if (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr) {
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr)
        dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      rescales = GetRescales(src0_scales, src1_scales, dst_scales, dst_->dtype(), append_eltwise_);
    } else {
      rescales = vector<float>(1, 1.f);
    }
    if (output_scale_ != 1.f) {
      for (int i = 0; i < rescales.size(); i++) {
        rescales[i] *= output_scale_;
      }
    }
    attr.set_output_scales(ic_dim, rescales);
  }
  dnnl::post_ops po;
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
  if (dst_->dtype() == "u8") {
    if (append_eltwise_) {
      float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
      po.append_eltwise(dst_scales[0], algorithm::eltwise_linear, 1., zero_point);
    } else {
      vector<int> dst_zero_points;
      dst_zero_points = GetZeroPoints(dst_min_->data(), dst_scales, dst_->dtype());
      attr.set_zero_points(DNNL_ARG_DST, ic_dim, dst_zero_points);
    }
  }
  if (binary_add_) {
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    binary_m_ = memory(binary_md, eng_);
  }
  attr.set_post_ops(po);
  matmul_pd_ = dnnl::matmul::primitive_desc(matmul_d, attr, eng_);

  // 2.4 Prepare primitive objects (cached)
  matmul_p_ = dnnl::matmul(matmul_pd_);

  // 2.5 Prepare memory objects (cached)
  src0_m_ = memory(src0_md, eng_);
  dst_m_ = memory(dst_md, eng_);
  if (has_bias) {
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
  // when change data value please use mutable_data
  if (post_ != nullptr && !binary_add_) {
    LOG(INFO) << "matmul has post op " << post_->name();
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
  auto dst_data = dst_->mutable_data();

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

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  if (!cache_weight_) memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;

  if (binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(reinterpret_cast<void*>(post_ptr), eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_m_;
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
}

REGISTER_OPERATOR_CLASS(Matmul);
}  // namespace executor
