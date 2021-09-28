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
#ifndef DEEP_ENGINE_EXECUTOR_PYTHON_PYBIND_TENSOR_HPP_
#define DEEP_ENGINE_EXECUTOR_PYTHON_PYBIND_TENSOR_HPP_


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <utility>
#include "executor.hpp"

namespace py = pybind11;

// type caster: executor::Tensor <-> NumPy-array
namespace pybind11 { namespace detail {
template <>
struct type_caster <executor::Tensor> {
 public:
    PYBIND11_TYPE_CASTER(executor::Tensor, _("executor::Tensor"));

    // Conversion part 1 (Python -> C++)
    bool load(py::handle src, bool convert) {
      if (!convert) {
        return false;
      }

      auto buf = py::array::ensure(src);

      if (!buf) {
        return false;
      }

      std::vector<int64_t> shape(buf.ndim());

      for (int i = 0 ; i < buf.ndim() ; i++) {
        shape[i] = buf.shape()[i];
      }
      string dtype = "fp32";
      if (py::isinstance<py::array_t<int32_t>>(buf)) dtype = "int32";
      if (py::isinstance<py::array_t<float>>(buf)) dtype = "fp32";
      if (py::isinstance<py::array_t<char>>(buf)) dtype = "s8";
      if (py::isinstance<py::array_t<unsigned char>>(buf)) dtype = "u8";

      value = executor::Tensor(const_cast<void*>(buf.data()), shape, dtype);

      return true;
    }

    // Conversion part 2 (C++ -> Python)
    static py::handle cast(const executor::Tensor& src,
      py::return_value_policy policy, py::handle parent) {
      py::array a;
      if (src.dtype() == "fp32") {
        a = py::array(std::move(src.shape()), reinterpret_cast<const float*>(src.raw_data()));
      } else if (src.dtype() == "int32") {
        a = py::array(std::move(src.shape()), reinterpret_cast<const int32_t*>(src.raw_data()));
      } else if (src.dtype() == "u8") {
        a = py::array(std::move(src.shape()), reinterpret_cast<const uint8_t*>(src.raw_data()));
      } else if (src.dtype() == "s8") {
        a = py::array(std::move(src.shape()), reinterpret_cast<const int8_t*>(src.raw_data()));
      }
      return a.release();
    }
};
}  // namespace detail
}  // namespace pybind11

#endif  // DEEP_ENGINE_EXECUTOR_PYTHON_PYBIND_TENSOR_HPP_
