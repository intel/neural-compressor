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
#include <climits>
#include <fstream>
#include <string>
#include <utility>

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    // https://stackoverflow.com/questions/18024672/what-registers-are-preserved-through-a-linux-x86-64-function-call
    // r12, r13, r14, r15, rbx, rsp, rbp are the callee-saved registers - they
    // have a "Yes" in the "Preserved across
    // function calls" column.
    // usually we use r12, r13, r14 for src.
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN32
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX), abi_param2(Xbyak::Operand::RDX),
    abi_param3(Xbyak::Operand::R8), abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI), abi_param2(Xbyak::Operand::RSI),
    abi_param3(Xbyak::Operand::RDX), abi_param4(Xbyak::Operand::RCX), abi_param5(Xbyak::Operand::R8),
    abi_param6(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RCX);
#endif

namespace jd {
class jit_generator : public Xbyak::CodeGenerator {
 public:
  explicit jit_generator(size_t code_size = MAX_CODE_SIZE, void* code_ptr = nullptr)
      : Xbyak::CodeGenerator(code_size, (code_ptr == nullptr) ? Xbyak::AutoGrow : code_ptr) {}
  virtual ~jit_generator() {}
  void dump_asm();  // print assembly code

 public:
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  template <typename... T>
  inline void operator()(T... args) const {
    using func_ptr = void (*)(const T... args);
    auto fptr = reinterpret_cast<func_ptr>(jit_ker_);
    (*fptr)(std::forward<T>(args)...);
  }
  virtual bool create_kernel();

  template <typename T>
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, T raw_offt, bool bcast = false);

  Xbyak::Address make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt, const Xbyak::Reg64& tmp_reg,
                                bool bcast = false);
  Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64& base, size_t raw_offt, const Xbyak::Reg64& reg_offt,
                                         bool bcast = false);

 protected:
  // derived jit_domain implementation
  virtual void generate() = 0;
  const uint8_t* get_code();

 protected:
  const uint8_t* jit_ker_ = nullptr;
  static constexpr uint64_t MAX_CODE_SIZE = 256 * 1024 * 1024;
  static constexpr int VEC = 16;  // 512 bits of ZMM register divided by S32 bits.
  int callee_functions_code_size_ = 0;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
