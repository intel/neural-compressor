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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
#include <string>
#include <unordered_map>
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"
#include "../sparse_operators/sparse_inner_product.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {

using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

// \brief InnerProduct or Batchmatmul operators
class InnerProductOperator : public Operator {
 public:
  explicit InnerProductOperator(const OperatorConfig& conf);
  virtual ~InnerProductOperator();

 public:
  // void ParseOperatorConfig();
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void PrepareDense(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void ReshapeSparse(const vector<Tensor*>& input, const vector<Tensor*>& output);
#if __AVX512F__
  void ForwardSparse(const vector<Tensor*>& input, const vector<Tensor*>& output);
#endif
  void PrepareSparse(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void DynamicForward(vector<float>* src0_compensation_ptr, vector<float>* rescales_ptr,
                      vector<float>* dynamic_bias_ptr, memory* any_bias_m_ptr);
  void RuntimeMinmax();
  void CalculateCompensation(const vector<int64_t>& src1_shape, const vector<int64_t>& src1_stride,
                             const vector<int64_t>& zero_point_stride);
  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  // The input tensors x and y are [..., r_x, c_x] and [..., r_y, c_y].
  // The output tensor is [..., r_o, c_o], where:
  // r_o = c_x if adj_x else r_x
  // c_o = r_y if adj_y else c_y
  // Matrix can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it).
  // So "adj_" decide the highest two dimensions, and is the built-in operation of InnerProduct OP.
  // While "perm" decide all dimensions, and is the external Trans OP. Both are transpose.
  bool weight_cached_;
  bool has_bias_;
  bool format_any_;
  bool append_sum_;
  bool binary_add_;
  bool gelu_erf_;
  bool gelu_tanh_;
  bool gelu_split_;
  bool tanh_;
  bool sigmoid_;
  bool relu_;

  bool append_eltwise_;
  bool is_dynamic_ = false;
  float output_scale_ = 1.f;
  vector<float> dst_scales_;
  string output_dtype_ = "fp32";
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  vector<int64_t> dst_perm_;
  vector<int64_t> compensation_;
  memory::desc scale_md_;
  memory::desc compensation_md_;

  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  dnnl::inner_product_forward::primitive_desc inner_product_pd_;
  dnnl::inner_product_forward inner_product_p_;
  unordered_map<int, memory> memory_args_;
  dnnl::primitive_attr attr_;

  dnnl::engine gelu_eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream gelu_eng_stream_ = dnnl::stream(gelu_eng_);
  dnnl::eltwise_forward::primitive_desc gelu_pd_;
  dnnl::eltwise_forward gelu_p_;
  unordered_map<int, memory> gelu_memory_args_;

  memory::desc src1_md_;
  memory::desc any_src1_md_;
  memory::desc bias_md_;
  memory::desc any_bias_md_;
  memory src0_m_;
  memory src1_m_;
  memory bias_m_;
  memory dst_m_;
  memory gelu_m_;
  memory binary_m_;

  Tensor* src0_ = nullptr;
  Tensor* src1_ = nullptr;
  Tensor* bias_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src0_min_ = nullptr;
  Tensor* src0_max_ = nullptr;

  Tensor* src1_min_ = nullptr;
  Tensor* src1_max_ = nullptr;

  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;

  bool dense_flag_ = false;
  float sparse_threshold_ = 0.7;
  float weight_zero_ratio_ = 0.0;

  BSCMatrix<float>* sparse_weight_ = nullptr;
  BSCMatrix<int8_t>* sparse_weight_int8_ = nullptr;
  vector<int64_t> blocksize_ = {1, 16};
  vector<float> rescales_;
  // M dimension num of per BLOCK is 4 for sparse kernel algorithm
  int64_t M_NBLK_ = 4;
  string append_op_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
