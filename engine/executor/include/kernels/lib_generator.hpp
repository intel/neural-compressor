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
#ifndef ENGINE_EXECUTOR_INCLUDE_KERNELS_LIB_GENERATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_KERNELS_LIB_GENERATOR_HPP_
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include "kernels/utils.hpp"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    // https://stackoverflow.com/questions/18024672/what-registers-are-preserved-through-a-linux-x86-64-function-call
    // r12, r13, r14, r15, rbx, rsp, rbp are the callee-saved registers - they have a "Yes" in the "Preserved across
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

namespace executor {
class LibGenerator : public Xbyak::CodeGenerator {
 public:
  explicit LibGenerator(size_t code_size = MAX_CODE_SIZE, void* code_ptr = nullptr, bool use_autogrow = true)
      : Xbyak::CodeGenerator(code_size, (code_ptr == nullptr && use_autogrow) ? Xbyak::AutoGrow : code_ptr) {}
  virtual ~LibGenerator() {}

 public:
  template <typename... kernel_args_t>
  inline void operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = void (*)(const kernel_args_t... args);
    auto* fptr = (jit_kernel_func_t)jit_ker_;
    (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  const uint8_t* get_kernel() const;

  virtual bool create_kernel() {
    generate();
    jit_ker_ = getCode();
    return (jit_ker_) ? true : false;
  }

 private:
  const Xbyak::uint8* getCode() {
    this->ready();
    if (!is_initialized()) return nullptr;
    const Xbyak::uint8* code = CodeGenerator::getCode();
    // TODO(hengyu): onednn use for profiling
    // register_jit_code(code, getSize());
    return code;
  }

  static inline bool is_initialized() { return Xbyak::GetError() == Xbyak::ERR_NONE; }

 public:
  Xbyak::Reg64 param1 = abi_param1;
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Xmm& x) { movdqu(addr, x); }

  void uni_vmovdqu(const Xbyak::Xmm& x, const Xbyak::Address& addr) { vmovdqu(x, addr); }

  void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Zmm& x) { vmovdqu32(addr, x); }

  void uni_vmovd(const Xbyak::Xmm& x, const Xbyak::Address& addr) { vmovd(x, addr); }

  void uni_vmovq(const Xbyak::Xmm& x, const Xbyak::Reg64& r) { vmovq(x, r); }

  void uni_vzeroupper() {
    // TODO(hengyu): handle non-avx case
    vzeroupper();
  }

  template <typename T>
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, T raw_offt, bool bcast = false) {
    using Xbyak::Address;
    using Xbyak::Reg64;
    using Xbyak::RegExp;
    using Xbyak::Zmm;

    assert(raw_offt <= INT_MAX);
    auto offt = static_cast<int>(raw_offt);

    int scale = 0;

    if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
      offt = offt - 2 * EVEX_max_8b_offt;
      scale = 1;
    } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
      offt = offt - 4 * EVEX_max_8b_offt;
      scale = 2;
    }

    auto re = RegExp() + base + offt;
    if (scale) re = re + reg_EVEX_max_8b_offt * scale;

    if (bcast)
      return zword_b[re];
    else
      return zword[re];
  }

  Xbyak::Address make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt, const Xbyak::Reg64& tmp_reg,
                                bool bcast = false) {
    if (offt > INT_MAX) {
      mov(tmp_reg, offt);
      return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
    } else {
      return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
    }
  }

  Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64& base, size_t raw_offt, const Xbyak::Reg64& reg_offt,
                                         bool bcast = false) {
    if (raw_offt > INT_MAX) {
      return make_safe_addr(base, raw_offt, reg_offt, bcast);
    } else {
      return EVEX_compress_addr(base, raw_offt, bcast);
    }
  }

  void safe_add(const Xbyak::Reg64& base, size_t raw_offt, const Xbyak::Reg64& reg_offt) {
    if (raw_offt > INT_MAX) {
      mov(reg_offt, raw_offt);
      add(base, reg_offt);
    } else {
      add(base, raw_offt);
    }
  }

  void safe_sub(const Xbyak::Reg64& base, size_t raw_offt, const Xbyak::Reg64& reg_offt) {
    if (raw_offt > INT_MAX) {
      mov(reg_offt, raw_offt);
      sub(base, reg_offt);
    } else {
      sub(base, raw_offt);
    }
  }

  // Disallow char-based labels completely
  void L(const char* label) = delete;
  // TODO(hengyu): onednn use reference here but cpplint won't allow
  void L(Xbyak::Label label) { Xbyak::CodeGenerator::L(label); }

  // TODO(hengyu): onednn use reference here but cpplint won't allow
  void L_aligned(Xbyak::Label label, int alignment = 16) {
    align(alignment);
    L(label);
  }
  void preamble() {
    // start of kernel function
    if (xmm_to_preserve) {
      sub(rsp, xmm_to_preserve * xmm_len);
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        uni_vmovdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(xmm_to_preserve_start + i));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    }
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
  }

  void postamble() {
    // end of kernel function
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    if (xmm_to_preserve) {
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        uni_vmovdqu(Xbyak::Xmm(xmm_to_preserve_start + i), ptr[rsp + i * xmm_len]);
      add(rsp, xmm_to_preserve * xmm_len);
    }
    uni_vzeroupper();
    ret();
  }

  void mul_by_const(const Xbyak::Reg& out, const Xbyak::Reg64& tmp, int value) {
    // Generates a shift + add sequence for multiplicating contents of the
    // out register by a known JIT-time value. Clobbers the tmp register.
    //
    // Pros compared to mul/imul:
    // - does not require using known registers
    // - not microcoded on Intel(R) Xeon Phi(TM) processors
    // Still, there are probably a lot of cases when mul/imul is faster on
    // Intel(R) Core(TM) processors. Not intended for critical path.

    // TODO(hengyu): detect when overflow is emminent (Roma)
    // TODO(hengyu): detect when using mul/imul is a better option (Roma)

    int p = 0;      // the current power of 2
    int old_p = 0;  // the last seen power of 2 such that value[old_p] != 0

    xor_(tmp, tmp);
    while (value) {
      if (value & 1) {
        int shift = p - old_p;
        if (shift) {
          shl(out, shift);
          old_p = p;
        }
        add(tmp, out);
      }
      value >>= 1;
      p++;
    }
    mov(out, tmp);
  }

  /*
    Saturation facility functions. enable to prepare the register
    holding the saturation upperbound and apply the saturation on
    the floating point register
   */
  template <typename Vmm>
  void init_saturate_f32_u8(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak::Reg64 reg_tmp, bool force_lbound = false) {
    assert(vmm_lbound.getIdx() != vmm_ubound.getIdx());

    auto init_vmm = [&](Vmm vmm, float value) {
      Xbyak::Xmm xmm_tmp(vmm.getIdx());
      mov(reg_tmp, float2int(value));
      uni_vmovq(xmm_tmp, reg_tmp);
      if (vmm.isYMM() || vmm.isZMM())
        vbroadcastss(vmm, xmm_tmp);
      else
        vshufps(vmm, xmm_tmp, xmm_tmp, 0);
    };

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen in store_data. The param force_lbound, will
    // force saturate values unconditionally to lbound.
    vpxord(vmm_lbound, vmm_lbound, vmm_lbound);

    const float saturation_ubound = static_cast<float>(UINT8_MAX);
    init_vmm(vmm_ubound, saturation_ubound);
  }

  template <typename Vmm>
  void init_saturate_f32_s32(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak::Reg64 reg_tmp, bool force_lbound = false) {
    assert(vmm_lbound.getIdx() != vmm_ubound.getIdx());

    auto init_vmm = [&](Vmm vmm, float value) {
      Xbyak::Xmm xmm_tmp(vmm.getIdx());
      mov(reg_tmp, float2int(value));
      uni_vmovq(xmm_tmp, reg_tmp);
      if (vmm.isYMM() || vmm.isZMM())
        vbroadcastss(vmm, xmm_tmp);
      else
        vshufps(vmm, xmm_tmp, xmm_tmp, 0);
    };

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen in store_data. The param force_lbound, will
    // force saturate values unconditionally to lbound.
    if (force_lbound) {
      const float saturation_lbound = INT32_MIN;
      init_vmm(vmm_lbound, saturation_lbound);
    }

    const float saturation_ubound = static_cast<float>(INT32_MAX);
    init_vmm(vmm_ubound, saturation_ubound);
  }
  template <typename Vmm>
  void init_saturate_f32_s8(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak::Reg64 reg_tmp, bool force_lbound = false) {
    assert(vmm_lbound.getIdx() != vmm_ubound.getIdx());

    auto init_vmm = [&](Vmm vmm, float value) {
      Xbyak::Xmm xmm_tmp(vmm.getIdx());
      mov(reg_tmp, float2int(value));
      uni_vmovq(xmm_tmp, reg_tmp);
      if (vmm.isYMM() || vmm.isZMM())
        vbroadcastss(vmm, xmm_tmp);
      else
        vshufps(vmm, xmm_tmp, xmm_tmp, 0);
    };

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen in store_data. The param force_lbound, will
    // force saturate values unconditionally to lbound.
    if (force_lbound) {
      const float saturation_lbound = INT8_MIN;
      init_vmm(vmm_lbound, saturation_lbound);
    }

    const float saturation_ubound = static_cast<float>(INT8_MAX);
    init_vmm(vmm_ubound, saturation_ubound);
  }

  template <typename Vmm>
  void saturate_f32_u8(const Vmm& vmm, const Vmm& vmm_lbound, const Vmm& vmm_ubound, bool force_lbound = false) {
    // This function is used to saturate to odt in f32 before converting
    // to s32 in order to avoid bad saturation due to cvtps2dq
    // behavior (it returns INT_MIN if the f32 is out of the
    // s32 range)
    vmaxps(vmm, vmm, vmm_lbound);
    vminps(vmm, vmm, vmm_ubound);
  }
  template <typename Vmm>
  void saturate_f32_s(const Vmm& vmm, const Vmm& vmm_lbound, const Vmm& vmm_ubound, bool force_lbound = false) {
    // This function is used to saturate to odt in f32 before converting
    // to s32 in order to avoid bad saturation due to cvtps2dq
    // behavior (it returns INT_MIN if the f32 is out of the
    // s32 range)
    if (force_lbound) {
      vmaxps(vmm, vmm, vmm_lbound);
    }
    vminps(vmm, vmm, vmm_ubound);
  }

 private:
// from: https://stackoverflow.com/questions/24299543/saving-the-xmm-register-before-function-call
#ifdef _WIN32
  // https://docs.microsoft.com/en-us/cpp/build/x64-software-conventions?redirectedfrom=MSDN&view=msvc-170
  // xmm6:xmm15 must be preserved as needed by caller
  const size_t xmm_to_preserve_start = 6;
  const size_t xmm_to_preserve = 10;
#else
  // https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI: page23
  // on Linux those are temporary registers, and therefore don't have to be preserved
  const size_t xmm_to_preserve_start = 0;
  const size_t xmm_to_preserve = 0;
#endif

  const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

  const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * xmm_len;

 protected:
  virtual void generate() = 0;
  const Xbyak::uint8* jit_ker_ = nullptr;

 protected:
  // current op kernel
  const uint8_t* ker_ = nullptr;
  static std::unordered_map<uint64_t, const uint8_t*> ker_cache_;
  static constexpr size_t MAX_CODE_SIZE = 256 * 1024;
  static constexpr size_t xmm_len = 16;  // 512 bits of ZMM register divided by S32 bits.
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_KERNELS_LIB_GENERATOR_HPP_
