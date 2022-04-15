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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
#include <string>
#include <fstream>
#include <utility>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace jd {
class jit_generator : public Xbyak::CodeGenerator {
 public:
  explicit jit_generator(size_t code_size = MAX_CODE_SIZE, void* code_ptr = nullptr)
  : Xbyak::CodeGenerator(code_size, (code_ptr == nullptr) ? Xbyak::AutoGrow : code_ptr) {}
  virtual ~jit_generator() {}
  void dump_asm();  // print assembly code

 public:
  template <typename... T>
  inline void operator()(T... args) const {
    using func_ptr = void (*)(const T... args);
    auto fptr = reinterpret_cast<func_ptr>(jit_ker_);
    (*fptr)(std::forward<T>(args)...);
  }
  virtual bool create_kernel();

 protected:
  // derived jit_domain implementation
  virtual void generate() = 0;
  const uint8_t* get_code();

 protected:
  const uint8_t* jit_ker_ = nullptr;
  static constexpr uint64_t MAX_CODE_SIZE = 256 * 1024;
  static constexpr int VEC = 16;  // 512 bits of ZMM register divided by S32 bits.
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
