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

#include "jit_generator.hpp"

namespace jd {
bool jit_generator::create_kernel() {
  generate();
  // dump_asm();
  jit_ker_ = get_code();
  return (jit_ker_ != nullptr);
}

const uint8_t* jit_generator::get_code() {
  this->ready();
  auto code = CodeGenerator::getCode();
  return code;
}

void jit_generator::dump_asm() {
  std::string file_name("temp.bin");
  std::ofstream out_file(file_name, std::ios::out | std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(getCode()), getSize());
  out_file.close();
  std::string cmd = "objdump -M x86-64 -D -b binary -m i386 " + file_name;
  system(cmd.c_str());
  remove(file_name.c_str());
}
}  // namespace jd
