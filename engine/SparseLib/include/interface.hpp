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

#ifndef ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
#include <omp.h>
#include <vector>
#include <cstdint>
#include <memory>
#include "param_types.hpp"
#include "operator_config.hpp"
#include "engine.hpp"
#include "cpu_engine.hpp"
#include "engine_factory.hpp"
#include "kernel_desc.hpp"
#include "kernel_framework.hpp"
#include "kernel_cache.hpp"
#include "utils.hpp"
#include "kernels/sparse_data.hpp"

namespace jd {
/**
 * @brief Proxy pattern. The proxy could interface to anything.
 *  Similar to onednn's "struct handle". oneapi/dnnl/dnnl.hpp:136.
 */
template<typename T, typename arg_t = void>
class proxy_base {
 public:
  proxy_base() {}
  virtual ~proxy_base() {}

 public:
  inline void reset_sp(const std::shared_ptr<const T>& sp) {
    data_handle_ = sp;
  }
  inline const std::shared_ptr<const T>& get_sp() const {
    return data_handle_;
  }

 protected:
  // internal functions of creat the proxy object.
  virtual bool create_proxy_object(std::shared_ptr<const T>& result_ref, const arg_t& arg) = 0;  // NOLINT

 private:
  std::shared_ptr<const T> data_handle_;
};

/**
 * @brief Base proxy class, interfacing to the real/cached kernel_desc_t.
 */
class kernel_desc_proxy : public proxy_base<kernel_desc_t, operator_config> {
 public:
  kernel_desc_proxy() {}
  explicit kernel_desc_proxy(const operator_config& op_cfg);
  virtual ~kernel_desc_proxy() {}

 protected:
  bool create_proxy_object(std::shared_ptr<const kernel_desc_t>& result_ref, const operator_config& op_cfg) override;

 public:
  inline const jd::kernel_kind& kernel_kind() const { return get_sp()->kernel_kind(); }

 protected:
  const std::vector<impl_list_item_t>* impl_list_ = nullptr;
};

/**
 * @brief Base proxy class, interfacing to the real/cached kernel_framework_t.
 */
class kernel_framework_proxy : public proxy_base<kernel_framework_t, std::shared_ptr<const kernel_desc_t>> {
 public:
  kernel_framework_proxy() {}
  explicit kernel_framework_proxy(const kernel_desc_proxy& kdp);
  virtual ~kernel_framework_proxy() {}

 protected:
  bool create_proxy_object(std::shared_ptr<const kernel_framework_t>& result_ref,
    const std::shared_ptr<const kernel_desc_t>& kd) override;

 public:
  inline const jd::kernel_kind& kernel_kind() const { return get_sp()->kd()->kernel_kind(); }
  void execute(const std::vector<const void*>& rt_data);
};


//// The following paragraphs are the various derived kernels and its descriptors.
/**
 * @brief Derived proxy class, interfacing to the real/cached sparse_matmul_desc_t.
 */
class sparse_matmul_desc : public kernel_desc_proxy {
 public:
  sparse_matmul_desc() {}
  explicit sparse_matmul_desc(const operator_config& op_cfg) : kernel_desc_proxy(op_cfg) {}
  virtual ~sparse_matmul_desc() {}
};

/**
 * @brief Derived proxy class, interfacing to the real/cached sparse_matmul_t.
 */
class sparse_matmul : public kernel_framework_proxy {
 public:
  sparse_matmul() {}
  explicit sparse_matmul(const kernel_desc_proxy& kdp) : kernel_framework_proxy(kdp) {}
  virtual ~sparse_matmul() {}
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
