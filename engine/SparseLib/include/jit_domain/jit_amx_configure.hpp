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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_CONFIGURE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_CONFIGURE_HPP_

#include "jit_generator.hpp"

namespace jd {

class jit_amx_config_t : public jit_generator {
 public:
  jit_amx_config_t() : jit_generator() {}
  virtual ~jit_amx_config_t() {}

  void tile_configure(void* palette) const { (*this)(palette); }

 private:
  void generate() override;
};

class jit_amx_release_t : public jit_generator {
 public:
  jit_amx_release_t() : jit_generator() {}
  virtual ~jit_amx_release_t() {}

  void tile_release() const { (*this); }

 private:
  void generate() override;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_CONFIGURE_HPP_
