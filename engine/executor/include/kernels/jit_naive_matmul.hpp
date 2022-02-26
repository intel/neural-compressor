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

#ifndef ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_MATMUL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_MATMUL_HPP_

#include <memory>
#include "kernels/jit_naive_gemm_kernel.hpp"

namespace executor {

struct jit_naive_matmul {
  jit_naive_matmul(int m_, int n_, int k_) {
    m = m_;
    n = n_;
    k = k_;
    kernel_.reset(new jit_naive_gemm_kernel(m, n, k));
    kernel_->create_kernel();
  }

 public:
  void operator()(naive_gemm_params_t args);

 private:
  std::unique_ptr<jit_naive_gemm_kernel> kernel_;
  int m;
  int n;
  int k;
};

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_MATMUL_HPP_
