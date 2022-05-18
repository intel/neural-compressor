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

#ifndef ENGINE_SPARSELIB_INCLUDE_TENSOR_DESC_HPP_
#define ENGINE_SPARSELIB_INCLUDE_TENSOR_DESC_HPP_
#include <functional>
#include <numeric>
#include <vector>

#include "param_types.hpp"

namespace jd {
class tensor_desc {
 public:
  tensor_desc() : shape_({}), dtype_(data_type::undef), ftype_(format_type::undef) {}
  tensor_desc(const std::vector<int64_t>& shape, const data_type& dtype, const format_type& ftype)
      : shape_(shape), dtype_(dtype), ftype_(ftype) {}
  virtual ~tensor_desc() {}

 public:
  bool operator==(const tensor_desc& rhs) const {
    return (shape_ == rhs.shape_) && (dtype_ == rhs.dtype_) && (ftype_ == rhs.ftype_);
  }

 public:
  inline const std::vector<int64_t>& shape() const { return shape_; }
  inline const data_type& dtype() const { return dtype_; }
  inline const format_type& ftype() const { return ftype_; }
  inline uint64_t size() const { return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<uint64_t>()); }

 private:
  std::vector<int64_t> shape_;
  data_type dtype_;
  format_type ftype_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_TENSOR_DESC_HPP_
