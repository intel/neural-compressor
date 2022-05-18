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

#pragma GCC push_options
#pragma GCC optimize("O0")
void sparselib_configure_tiles() {
  tileconfig_t sparselib_tc;
  // Filling tile configure structure. Could be done offline.
  sparselib_tc.palette_id = 1;
  // zeros reserved
  for (int t = 0; t < 15; ++t) {
    sparselib_tc.reserved[t] = 0;
  }
  // Configure C tiles
  for (int t = 0; t < 4; ++t) {
    sparselib_tc.rows[t] = 16;
    sparselib_tc.colb[t] = 64;
  }
  // Configure A tiles
  for (int t = 4; t < 6; ++t) {
    sparselib_tc.rows[t] = 16;
    sparselib_tc.colb[t] = 64;
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (int t = 6; t < 8; ++t) {
    sparselib_tc.rows[t] = 16;
    sparselib_tc.colb[t] = 64;
  }
  // Zeroing other cols & rows
  for (int t = 8; t < 16; ++t) {
    sparselib_tc.rows[t] = 0;
    sparselib_tc.colb[t] = 0;
  }
  _tile_loadconfig(&sparselib_tc);
}
#pragma GCC pop_options
