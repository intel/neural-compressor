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

#include "i_malloc.hpp"

#include <fcntl.h>
#include <nmmintrin.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tmmintrin.h>
#include <unistd.h>
#include <xmmintrin.h>

#include <unordered_map>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Global memory pool instance
static struct malloc_mempool g_mempool;

void mempool_dump() {
  struct malloc_elem* elem = g_mempool.free_head;
  printf("Free list:\n");
  while (elem) {
    char* end = reinterpret_cast<char*>(elem) + sizeof(struct malloc_elem) + elem->size;
    printf("\t%p-%p:%lu\n", elem, end, elem->size);
    struct malloc_elem* p = elem;
    elem = elem->next_free;
    if (elem != NULL && elem->prev_free != p) {
      printf("ERROR: the free element list is not correct.\n");
      exit(-1);
    }
  }
}

// Memory pool initialization
static void mempool_init(struct malloc_mempool* pool, size_t pool_size) {
  int memsize = pool_size;
#ifdef DEBUG
  printf("before mmap, size=%lu\n", pool_size);
#endif
  void* ptr = mmap(0, memsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef DEBUG
  printf("after mmap, ptr=%p, errno=%d\n", ptr, errno);
#endif
  if (ptr == reinterpret_cast<void*>(-1)) {
    printf("ERROR: Cannot allocate the memory pool, will fall back to glibc.\n");
    pool->start_addr = NULL;
    pool->free_head = NULL;
    pool->alloc_count = 0;
    pool->total_size = 0;
    pool->initialized = 1;
    pool->max_unstsfd_size = 0;
    return;
  }

  struct malloc_elem* elem = (struct malloc_elem*)ptr;

  elem->next_free = NULL;
  elem->prev_free = NULL;
  elem->prev = NULL;
  elem->next = NULL;
  elem->size = memsize - sizeof(struct malloc_elem);
  elem->free_time = 0;
  elem->hot = HOT_HEAD;
  elem->state = ELEM_FREE;

  pool->start_addr = ptr;
  pool->free_head = elem;
  pool->alloc_count = 0;
  pool->total_size = memsize;
  pool->initialized = 1;
  pool->max_unstsfd_size = 0;
}

static void mempool_enlarge(struct malloc_mempool* pool, size_t increase_size) {
  if (pool->start_addr != NULL) {
    int ret = munmap(pool->start_addr, pool->total_size);
    if (ret != 0) {
      printf("Failed to unmap the memory.\n");
    }
  }
  mempool_init(pool, pool->total_size + increase_size);
}

// Find the available element by status
// Return the first free one, which is freed most recently
static struct malloc_elem* mempool_find_element(struct malloc_mempool* pool, size_t size) {
  struct malloc_elem* elem = pool->free_head;

  while (elem) {
    // Return the first one once the size is big enough
    if (elem->size >= size) {
      return elem;
    }

    elem = elem->next_free;
  }

  return NULL;
}

static struct malloc_elem* mempool_do_alloc(struct malloc_elem* elem, size_t size) {
  // Wasted is acceptable
  if (elem->size - size < MAX_WASTED_SIZE) {
    struct malloc_elem* prev = elem->prev_free;
    struct malloc_elem* next = elem->next_free;
    if (prev != NULL) {
      prev->next_free = elem->next_free;
    } else {
      g_mempool.free_head = elem->next_free;
    }
    if (next != NULL) {
      next->prev_free = prev;
    }
    elem->state = ELEM_BUSY;
    return elem;
  }

  // Need to split the element
  size_t alloc_size = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;

  struct malloc_elem* new_elem =
      (struct malloc_elem*)(reinterpret_cast<char*>(elem) + sizeof(struct malloc_elem) + alloc_size);
  new_elem->next_free = elem->next_free;
  new_elem->prev_free = elem->prev_free;
  new_elem->prev = elem;
  new_elem->next = elem->next;
  new_elem->size = elem->size - alloc_size - sizeof(struct malloc_elem);
  new_elem->free_time = elem->free_time;
  new_elem->hot = elem->hot;
  new_elem->state = ELEM_FREE;

  if (elem->next != NULL) {
    elem->next->prev = new_elem;
  }

  elem->next = new_elem;
  elem->size = alloc_size;
  elem->state = ELEM_BUSY;

  // Re-chain the free elements
  if (elem->prev_free != NULL) {
    elem->prev_free->next_free = new_elem;
  } else {
    g_mempool.free_head = new_elem;
  }
  if (elem->next_free != NULL) {
    elem->next_free->prev_free = new_elem;
  }

  return elem;
}

// Return 1 if it is inside our memory pool, otherwise -1
static int check_addr(void* addr) {
  if (addr > g_mempool.start_addr && addr < g_mempool.start_addr + g_mempool.total_size) {
    return 1;
  }
  return -1;
}

inline static void* mempool_malloc(size_t size) {
  if (unlikely(g_mempool.initialized == 0)) {
    mempool_init(&g_mempool, MEMPOOL_INIT_SIZE);
    g_mempool.initialized = 1;
  }

  if (unlikely(g_mempool.start_addr == NULL)) {
    return NULL;
  }

  struct malloc_elem* elem = mempool_find_element(&g_mempool, size);
  if (likely(elem != NULL)) {
    elem = mempool_do_alloc(elem, size);
    g_mempool.alloc_count++;
  }

  if (likely(elem != NULL)) {
    void* addr = reinterpret_cast<void*>(reinterpret_cast<char*>(elem) + sizeof(struct malloc_elem));
    return addr;
  }

  return NULL;
}

void* i_malloc(size_t size) {
#ifdef DEBUG
  printf("MALLOC: size=%lu\n", size);
#endif

  void* addr = mempool_malloc(size);

  if (unlikely(addr == NULL)) {
    addr = malloc(size);
    if (size > g_mempool.max_unstsfd_size) {
      g_mempool.max_unstsfd_size = size;
    }
#ifdef DEBUG
    printf("Memory allocation failed, size=%lu\n", size);
#endif
  }

#ifdef DEBUG
  mempool_dump();
#endif

  return addr;
}

static void dlist_move_to_head(struct malloc_mempool* pool, struct malloc_elem* e) {
  struct malloc_elem* n = e->next_free;
  struct malloc_elem* p = e->prev_free;
  // as e is not the head, p is always not NULL
  if (p != NULL) {
    p->next_free = n;
  }
  if (n != NULL) {
    n->prev_free = p;
  }
  e->next_free = pool->free_head;
  e->prev_free = NULL;
  pool->free_head->prev_free = e;
  pool->free_head = e;
}

static void dlist_remove_node(struct malloc_mempool* pool, struct malloc_elem* e) {
  struct malloc_elem* n = e->next_free;
  struct malloc_elem* p = e->prev_free;

  if (p != NULL) {  // p is always not NULL
    p->next_free = n;
  }
  if (n != NULL) {
    n->prev_free = p;
  }

  if (pool->free_head == e) {
    pool->free_head = n;
  }
}

void i_free(void* ptr) {
  if (unlikely(ptr == NULL)) {
    return;
  }

#ifdef DEBUG
  printf("FREE: %p\n", reinterpret_cast<char*>(ptr) - sizeof(struct malloc_elem));
#endif

  int ret = check_addr(ptr);
  time_t now = time(NULL);

  // Inside our memory pool
  if (ret == 1) {
    struct malloc_elem* elem = (struct malloc_elem*)(reinterpret_cast<char*>(ptr) - sizeof(struct malloc_elem));

    struct malloc_elem* elem_prev = elem->prev;
    struct malloc_elem* elem_next = elem->next;
    if (elem_prev != NULL && elem_prev->state == ELEM_FREE) {
      // Both previous and next element should be merged
      if (elem_next != NULL && elem_next->state == ELEM_FREE) {
#ifdef DEBUG
        printf("Merge with both previous and next free node\n");
#endif
        elem_prev->next = elem_next->next;
        if (elem_next->next != NULL) {
          elem_next->next->prev = elem_prev;
        }
        elem_prev->free_time = now;
        elem_prev->hot = HOT_NONE;
        elem_prev->size += 2 * sizeof(struct malloc_elem) + elem->size + elem_next->size;
        dlist_remove_node(&g_mempool, elem_next);
        if (g_mempool.free_head != elem_prev) {
          dlist_move_to_head(&g_mempool, elem_prev);
        }
      } else {  // Only need to merge with the previous element
#ifdef DEBUG
        printf("Merge with previous free node\n");
#endif
        elem_prev->size += sizeof(struct malloc_elem) + elem->size;
        elem_prev->next = elem->next;
        elem_prev->free_time = now;
        elem_prev->hot = HOT_TAIL;
        if (elem->next != NULL) {
          elem->next->prev = elem_prev;
        }
        // Make the merged node as the head (as it is the hottest)
        if (g_mempool.free_head != elem_prev) {
          dlist_move_to_head(&g_mempool, elem_prev);
        }
      }
    } else {
      // Only need to merge with the next element
      if (elem_next != NULL && elem_next->state == ELEM_FREE) {
#ifdef DEBUG
        printf("Merge with next free node, e=%p\n", elem);
#endif
        elem->next_free = elem_next->next_free;
        elem->prev_free = elem_next->prev_free;
        if (elem->next_free != NULL) {
          elem->next_free->prev_free = elem;
        }
        if (elem->prev_free != NULL) {
          elem->prev_free->next_free = elem;
        }

        elem->next = elem_next->next;
        if (elem_next->next != NULL) {
          elem_next->next->prev = elem;
        }

        elem->size += sizeof(struct malloc_elem) + elem_next->size;
        elem->free_time = now;
        elem->hot = HOT_HEAD;
        elem->state = ELEM_FREE;

        // Make the merged node as the head (as it is the hottest)
        if (g_mempool.free_head != elem_next) {
          dlist_move_to_head(&g_mempool, elem);
        } else {
          g_mempool.free_head = elem;
        }
      } else {  // No element need to be merged
#ifdef DEBUG
        printf("Do not need to merge\n");
#endif
        elem->next_free = g_mempool.free_head;
        if (elem->next_free != NULL) {
          elem->next_free->prev_free = elem;
        }
        elem->prev_free = NULL;
        elem->free_time = now;
        elem->hot = HOT_NONE;
        elem->state = ELEM_FREE;

        g_mempool.free_head = elem;
      }
    }
  } else {  // Free the buffer allocated by malloc
    free(ptr);
  }

  // Need extra space to satisfy the memory requirement
  // and all allocated memory in memory pool got freed
  if (g_mempool.max_unstsfd_size > 0 && g_mempool.free_head != NULL &&
      g_mempool.free_head->size + sizeof(struct malloc_elem) == g_mempool.total_size) {
    const size_t page_size = 4 * 1024;
    size_t extra_size = (g_mempool.max_unstsfd_size + MAX_WASTED_SIZE + page_size - 1) / page_size * page_size;
#ifdef DEBUG
    printf("Enlarge the memory pool, extra_size=%lu\n", extra_size);
#endif
    mempool_enlarge(&g_mempool, extra_size);
  }

#ifdef DEBUG
  mempool_dump();
#endif
}
