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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "executor.hpp"
#include "pybind_tensor.hpp"
#include "tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(engine_py, m) {
  m.doc() = "pybind11 engine plugin";
  py::class_<executor::Model>(m, "Model")
      .def(py::init<std::string, std::string>())
      .def(py::init<executor::ModelConfig, std::string>())
      .def("forward", &executor::Model::Forward, py::arg("input"));

  py::class_<executor::TensorConfig>(m, "tensor_config")
      .def(py::init<std::string, const std::vector<int64_t>&, std::string, const std::vector<int64_t>&,
                    const std::vector<int64_t>&>());

  py::class_<executor::AttrConfig>(m, "attrs_config").def(py::init<const std::map<std::string, std::string>&>());

  py::class_<executor::OperatorConfig>(m, "op_config")
      .def(py::init<std::string, std::string, const std::vector<executor::TensorConfig*>&,
                    const std::vector<executor::TensorConfig*>&, executor::AttrConfig*>());

  py::class_<executor::ModelConfig>(m, "model_config")
      .def(py::init<std::string, const std::vector<executor::OperatorConfig*>&>())
      .def(py::init<YAML::Node>());
}
