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

#ifndef ENGINE_SPARSELIB_INCLUDE_AMX_UTILS_HPP_
#define ENGINE_SPARSELIB_INCLUDE_AMX_UTILS_HPP_
#include <immintrin.h>
#include <mutex>  // NOLINT
#include <cstdint>

#include "jit_domain/jit_amx_configure.hpp"

class tile_param_t {
 public:
  int TILE_M;
  int TILE_N;
  int TILE_K;
  bool is_bf16;
  int KPACK;

 public:
  bool operator!=(const tile_param_t& rhs) {
    return (TILE_M != rhs.TILE_M) | (TILE_K != rhs.TILE_K) | (TILE_N != rhs.TILE_N) | (is_bf16 != rhs.is_bf16) |
           (KPACK != rhs.KPACK);
  }
};

// Tile configure structure
struct tileconfig_t {
  uint8_t palette_id;
  uint8_t reserved[15];
  uint16_t colb[16];
  uint8_t rows[16];
};

void sparselib_configure_tiles(tile_param_t param, tileconfig_t* sparselib_tc);

/**
 * The amx_tile_config_t is in amx_tile_config_t mode to ensure all primitive share the
 * same configure. defines the `GetInstance` method that serves as an
 * alternative to constructor and lets clients access the same instance of this
 * class over and over.
 */
class amx_tile_config_t {
 private:
  static amx_tile_config_t* atc_instance_;
  static std::mutex mutex_;  // for thread safety

 protected:
  amx_tile_config_t() {
    tilecfg.create_kernel();
    tilerls.create_kernel();
  }
  ~amx_tile_config_t() {}
  tile_param_t param_ = {0};
  tileconfig_t* config_ = new tileconfig_t({0});

 public:
  /**
   * amx_tile_config_ts should not be cloneable.
   */
  amx_tile_config_t(amx_tile_config_t& other) = delete;  // NOLINT
  /**
   * amx_tile_config_ts should not be assignable.
   */
  void operator=(const amx_tile_config_t&) = delete;
  /**
   * This is the static method that controls the access to the singleton
   * instance. On the first run, it creates a singleton object and places it
   * into the static field. On subsequent runs, it returns the client existing
   * object stored in the static field.
   */

  static amx_tile_config_t* GetInstance();
  /**
   * Finally, any singleton should define some business logic, which can be
   * executed on its instance.
   */
  bool amx_tile_configure(tile_param_t param);
  bool amx_tile_release();
  jd::jit_amx_config_t tilecfg;
  jd::jit_amx_release_t tilerls;
};

#endif  // ENGINE_SPARSELIB_INCLUDE_AMX_UTILS_HPP_
