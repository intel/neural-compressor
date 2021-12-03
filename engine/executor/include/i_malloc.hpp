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

#ifndef ENGINE_EXECUTOR_INCLUDE_I_MALLOC_HPP_
#define ENGINE_EXECUTOR_INCLUDE_I_MALLOC_HPP_

#include <stddef.h>
#include <time.h>

#include <cstdint>

// Initial pool size in MB
#define MEMPOOL_INIT_SIZE_MB 16
#define MEMPOOL_INIT_SIZE (MEMPOOL_INIT_SIZE_MB * 1024 * 1024)

// Alignment (bytes boundary for the allocated address)
#define ALIGNMENT 64

// Acceptable wasted memory size
#define MAX_WASTED_SIZE 10240

struct malloc_elem;
struct malloc_mempool {
  void* start_addr;
  struct malloc_elem* free_head;
  unsigned initialized;
  unsigned alloc_count;
  size_t total_size;
  // Maximum unsatisfied size during the allocation
  size_t max_unstsfd_size;
};

enum elem_state { ELEM_FREE = 0, ELEM_BUSY };

enum elem_hot_status {
  HOT_NONE = 0,
  HOT_HEAD,  // Head is more hot, prefer to allocate head
  HOT_TAIL
};

struct malloc_elem {
  // Link the free buffers
  struct malloc_elem* next_free;
  struct malloc_elem* prev_free;
  // Always point to the previous/next element in physical layout
  struct malloc_elem* prev;
  struct malloc_elem* next;
  // Size not include malloc_elem itself
  size_t size;
  // Last free time
  time_t free_time;
  enum elem_hot_status hot;
  enum elem_state state;
  uint64_t guard;
};

void* i_malloc(size_t size);
void i_free(void* ptr);

#endif  // ENGINE_EXECUTOR_INCLUDE_I_MALLOC_HPP_
