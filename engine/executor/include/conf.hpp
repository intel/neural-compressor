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

#ifndef ENGINE_EXECUTOR_INCLUDE_CONF_HPP_
#define ENGINE_EXECUTOR_INCLUDE_CONF_HPP_

#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "yaml-cpp/yaml.h"
using std::map;
using std::string;
using std::vector;

namespace executor {

/**
 * @brief we can construct the config instance from yaml file or just from
 *        the element needed.
 */

class AttrConfig {
 public:
  explicit AttrConfig(const YAML::Node node) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      YAML::Node key = it->first;
      YAML::Node value = it->second;
      attrs_.insert({key.as<std::string>(), value.as<std::string>()});
    }
  }
  explicit AttrConfig(const map<string, string>& attr) : attrs_(attr) {}
  inline const map<string, string>& attributes() const { return attrs_; }

 protected:
  map<string, string> attrs_;
};

class TensorConfig {
 public:
  TensorConfig(const string& name, const YAML::Node& node)
      : name_(name), dtype_("fp32"), strides_({}), count_(0), location_({}) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      YAML::Node key = it->first;
      YAML::Node value = it->second;
      if (key.as<std::string>() == "shape") {
        shape_ = value.as<vector<int64_t>>();
      }
      if (key.as<std::string>() == "dtype") {
        dtype_ = value.as<std::string>();
      }
      if (key.as<std::string>() == "strides") {
        strides_ = value.as<vector<int64_t>>();
      }
      if (key.as<std::string>() == "location") {
        location_ = value.as<vector<int64_t>>();
      }
    }
  }

  TensorConfig(const string& name, const vector<int64_t>& shape = {}, const string& dtype = "fp32",
               const vector<int64_t>& strides = {}, const vector<int64_t>& location = {})
      : name_(name), shape_(shape), dtype_(dtype), strides_(strides), count_(0), location_(location) {}

  inline const string& name() const { return name_; }
  inline const vector<int64_t>& shape() const { return shape_; }
  inline const vector<int64_t>& location() const { return location_; }
  inline const string& dtype() const { return dtype_; }
  inline const vector<int64_t>& strides() const { return strides_; }
  inline void set_shape(const vector<int64_t>& shape) { shape_ = shape; }

 protected:
  string name_;
  int64_t count_;
  vector<int64_t> shape_;
  string dtype_;
  vector<int64_t> strides_;
  // (TODO) not good to add location in TensorConfig, it's used for weight tensor parse
  vector<int64_t> location_;
};

class OperatorConfig {
 public:
  OperatorConfig(const string name, const YAML::Node& node)
      : name_(name), attrs_(new AttrConfig(std::map<string, string>({}))) {
    for (auto v = node.begin(); v != node.end(); ++v) {
      YAML::Node key = v->first;
      YAML::Node value = v->second;
      if (key.as<std::string>() == "type") {
        type_ = value.as<std::string>();
        // (TODO) check the type is our supported
      }
      if (key.as<std::string>() == "input") {
        ParseTensor(value, &inputs_);
      }
      if (key.as<std::string>() == "output") {
        ParseTensor(value, &outputs_);
      }
      if (key.as<std::string>() == "attr") {
        attrs_ = new AttrConfig(value);
      }
    }
  }
  OperatorConfig(const string& name, const string& type, const vector<TensorConfig*>& inputs,
                 const vector<TensorConfig*>& outputs, AttrConfig* attrs)
      : name_(name), type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  void ParseTensor(const YAML::Node& node, vector<TensorConfig*>* tensors) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      auto name = it->first.as<std::string>();
      tensors->push_back(new TensorConfig(name, it->second));
    }
  }

 public:
  inline const string& name() const { return name_; }
  inline const string& type() const { return type_; }
  inline int input_tensor_size() const { return inputs_.size(); }
  inline int output_tensor_size() const { return outputs_.size(); }
  inline TensorConfig* input_tensors(int i = 0) const { return inputs_[i]; }
  inline TensorConfig* output_tensors(int i = 0) const { return outputs_[i]; }
  inline const map<string, string>& attributes() const { return attrs_->attributes(); }

 protected:
  string name_;
  string type_;
  vector<TensorConfig*> inputs_;
  vector<TensorConfig*> outputs_;
  AttrConfig* attrs_;
};

class ModelConfig {
 public:
  explicit ModelConfig(const YAML::Node& node) {
    YAML::Node model_config = node["model"];
    // iter out all the config and initialize
    for (auto it = model_config.begin(); it != model_config.end(); ++it) {
      YAML::Node key = it->first;
      YAML::Node value = it->second;
      if (key.as<std::string>() == "name") {
        name_ = value.as<std::string>();
      }
      if (key.as<std::string>() == "operator") {
        auto operator_config = model_config["operator"];
        ParseOperator(operator_config, operators_);
      }
    }
  }
  explicit ModelConfig(const string& conf_file) : ModelConfig(ParseConfig(conf_file)) {}

  ModelConfig(const string& name, const vector<OperatorConfig*>& operators) : name_(name), operators_(operators) {}

 public:
  void ParseOperator(const YAML::Node& node, vector<OperatorConfig*>& operators) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      auto name = it->first.as<std::string>();
      operators.push_back(new OperatorConfig(name, it->second));
    }
  }

  YAML::Node ParseConfig(const string& conf_file) { return YAML::LoadFile(conf_file); }
  bool CheckConfig() { return true; }

  inline const string& name() const { return name_; }
  inline const vector<OperatorConfig*>& operators() const { return operators_; }

 protected:
  string name_;
  vector<OperatorConfig*> operators_;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_CONF_HPP_
