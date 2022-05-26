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

#include "amx_utils.hpp"

amx_tile_config_t* amx_tile_config_t::atc_instance_{nullptr};
std::mutex amx_tile_config_t::mutex_;
static const jd::jit_amx_config_t tilecfg;
static const jd::jit_amx_release_t tilerls;

amx_tile_config_t* amx_tile_config_t::GetInstance() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (atc_instance_ == nullptr) {
    atc_instance_ = new amx_tile_config_t();
  }
  return atc_instance_;
}

bool amx_tile_config_t::amx_tile_configure(tile_param_t param) {
  if (param != param_) {
    param_ = param;
    sparselib_configure_tiles(param, config_);
    tilecfg.tile_configure(reinterpret_cast<void*>(config_));
  }
  return true;
}

bool amx_tile_config_t::amx_tile_release() {
  tilerls.tile_release();
  return true;
}

#pragma GCC push_options
#pragma GCC optimize("O0")
void sparselib_configure_tiles(tile_param_t param, tileconfig_t* sparselib_tc) {
  // Filling tile configure structure. Could be done offline.
  sparselib_tc->palette_id = 1;
  int sizeof_src_dtype = 1;
  if (param.is_bf16) {
    sizeof_src_dtype = 2;
  }
  int sizeof_dst_dtype = 4;
  // zeros reserved
  for (int t = 0; t < 15; ++t) {
    sparselib_tc->reserved[t] = 0;
  }
  // Configure C tiles
  for (int t = 0; t < 4; ++t) {
    sparselib_tc->rows[t] = param.TILE_M;
    sparselib_tc->colb[t] = param.TILE_N * sizeof_dst_dtype;
  }
  // Configure A tiles
  for (int t = 4; t < 6; ++t) {
    sparselib_tc->rows[t] = param.TILE_M;
    sparselib_tc->colb[t] = param.TILE_K * sizeof_src_dtype;
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (int t = 6; t < 8; ++t) {
    sparselib_tc->rows[t] = param.TILE_K / param.KPACK;
    sparselib_tc->colb[t] = param.TILE_N * param.KPACK * sizeof_src_dtype;
  }
  // Zeroing other cols & rows
  for (int t = 8; t < 16; ++t) {
    sparselib_tc->rows[t] = 0;
    sparselib_tc->colb[t] = 0;
  }
}
#pragma GCC pop_options
