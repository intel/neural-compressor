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

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../executor/include/i_malloc.hpp"
#include "gtest/gtest.h"

struct TestParams {
  std::vector<int> args;
};

static bool HasOverlap(const std::unordered_map<int, void*>& allocate_addrs) {
  std::vector<std::pair<void*, int> > addresses;
  addresses.reserve(allocate_addrs.size());
  for (auto it : allocate_addrs) {
    addresses.push_back(std::make_pair(it.second, it.first));
  }
  std::sort(addresses.begin(), addresses.end(),
            [](std::pair<void*, int>& a, std::pair<void*, int>& b) { return a.first < b.first; });

  std::pair<void*, int> pre = addresses[0];
  for (size_t i = 1; i < addresses.size(); ++i) {
    // Start address should > previous end address
    if (reinterpret_cast<char*>(addresses[i].first) <= reinterpret_cast<char*>(pre.first) + pre.second) {
      return true;
    }
    pre = addresses[i];
  }

  return false;
}

bool CheckResult(const TestParams& t) {
  const std::vector<int>& seqs = t.args;
  std::unordered_map<int, void*> allocate_addrs;
  bool ret = true;

  // Record the beginning address
  void* beggin_addr = i_malloc(64);
  i_free(beggin_addr);

  // Allocate/free according to the sequences
  for (int size : seqs) {
    auto it = allocate_addrs.find(size);
    if (it == allocate_addrs.end()) {
      void* ptr = i_malloc(size);
      allocate_addrs.insert(std::make_pair(size, ptr));
      if (HasOverlap(allocate_addrs)) {
        ret = false;
      }
    } else {
      i_free(it->second);
      allocate_addrs.erase(size);
    }
  }

  // Check if all the buffer now available
  void* addr = i_malloc(MEMPOOL_INIT_SIZE - sizeof(struct malloc_elem));
  i_free(addr);

  if (addr != beggin_addr) {
    ret = false;
  }

  return ret;
}

class IMallocTest : public testing::TestWithParam<TestParams> {
 protected:
  IMallocTest() {}
  ~IMallocTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(IMallocTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

// Return a sequence like 10, 20, 10, 20,
// which means to action sequences: alloc 10 bytes, alloc 20, free 10, free 20
static std::vector<int> GenerateAllocFreeSeqs(int alloc_num) {
  std::vector<int> action_sequences;
  std::vector<int> allocate_sizes;

  action_sequences.reserve(alloc_num * 2);
  allocate_sizes.reserve(alloc_num);

  for (int i = 0; i < alloc_num; ++i) {
    action_sequences.push_back(i);
    action_sequences.push_back(i);

    // Size between 10k and 100k, make sure it is different
    int size = 10 * 1024 + random() % (90 * 1024);
    while (std::find(allocate_sizes.begin(), allocate_sizes.end(), size) != allocate_sizes.end()) {
      size = 10 * 1024 + random() % (90 * 1024);
    }

    allocate_sizes.push_back(size);
  }

  // Shuffle it
  for (int i = 0; i < alloc_num * 2; ++i) {
    int j = i + random() % (alloc_num * 2 - i);
    // Swap element at i and j
    std::swap(action_sequences[i], action_sequences[j]);
  }

  // Place the size in the sequence
  for (int i = 0; i < alloc_num * 2; ++i) {
    int idx = action_sequences[i];
    action_sequences[i] = allocate_sizes[idx];
  }

  return action_sequences;
}

static auto GenerateCases = []() {
  std::vector<TestParams> cases;

  cases.push_back({GenerateAllocFreeSeqs(8)});
  cases.push_back({GenerateAllocFreeSeqs(32)});
  cases.push_back({GenerateAllocFreeSeqs(100)});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, IMallocTest, GenerateCases());
