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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "conf.hpp"
#include "operator.hpp"

namespace executor {

class Operator;

class OperatorRegistry {
 public:
  typedef shared_ptr<Operator> (*Creator)(const OperatorConfig&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    LOG(INFO) << "Gonna register " << type << "....";
    CHECK_EQ(registry.count(type), 0) << "Operator type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a operator using a OperatorConfig.
  static shared_ptr<Operator> CreateOperator(const OperatorConfig& conf) {
    const string& type = conf.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown operator type: " << type
                                      << " (known types: " << OperatorTypeListString() << ")";
    return registry[type](conf);
  }

  static vector<string> OperatorTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> operator_types;
    for (typename CreatorRegistry::iterator iter = registry.begin(); iter != registry.end(); ++iter) {
      operator_types.push_back(iter->first);
    }
    return operator_types;
  }

 private:
  // Operator registry should never be instantiated - everything is done with its
  // static variables.
  OperatorRegistry() {}

  static string OperatorTypeListString() {
    vector<string> operator_types = OperatorTypeList();
    string operator_types_str;
    for (vector<string>::iterator iter = operator_types.begin(); iter != operator_types.end(); ++iter) {
      if (iter != operator_types.begin()) {
        operator_types_str += ", ";
      }
      operator_types_str += *iter;
    }
    return operator_types_str;
  }
};

class OperatorRegisterer {
 public:
  OperatorRegisterer(const string& type, shared_ptr<Operator> (*creator)(const OperatorConfig&)) {
    OperatorRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_OPERATOR_CREATOR(type, creator) static OperatorRegisterer g_creator_##type(#type, creator);

#define REGISTER_OPERATOR_CLASS(type)                                         \
  shared_ptr<Operator> Creator_##type##Operator(const OperatorConfig& conf) { \
    return shared_ptr<Operator>(new type##Operator(conf));                    \
  }                                                                           \
  REGISTER_OPERATOR_CREATOR(type, Creator_##type##Operator)

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_
