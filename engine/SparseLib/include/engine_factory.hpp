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

#ifndef ENGINE_SPARSELIB_INCLUDE_ENGINE_FACTORY_HPP_
#define ENGINE_SPARSELIB_INCLUDE_ENGINE_FACTORY_HPP_
#include <memory>
#include <unordered_map>

#include "cpu_engine.hpp"
#include "param_types.hpp"

namespace jd {
class engine_factory {
 public:
  static engine_factory& instance() {
    static engine_factory inst;
    return inst;
  }
  const engine* create(const engine_kind& eng_kind) {
    const auto& it = mp_.find(eng_kind);
    if (it != mp_.end()) {
      return (*(it->second))();
    } else {
      return nullptr;
    }
  }

 private:
  void register_class() {
    if (!mp_.count(engine_kind::cpu)) {
      mp_[engine_kind::cpu] = &engine_factory::create_cpu_engine;
    }
  }
  static const engine* create_cpu_engine() {
    static std::shared_ptr<const cpu_engine> obj = std::make_shared<const cpu_engine>();
    return reinterpret_cast<const engine*>(obj.get());
  }

 private:
  engine_factory() { register_class(); }
  using create_fptr = const engine* (*)();
  std::unordered_map<engine_kind, create_fptr> mp_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_ENGINE_FACTORY_HPP_
