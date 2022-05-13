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

#ifndef ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "i_malloc.hpp"

namespace executor {
using std::map;
using std::shared_ptr;
using std::string;
using std::vector;
namespace ipc = boost::interprocess;
class MemoryAllocator {
 public:
  // vector will contain the used counts and size of a memory
  typedef std::map<void*, vector<size_t>> MemoryBuffer;
  typedef std::map<void*, string> BufferName;
  typedef std::map<string, bool> StrategyList;
  typedef std::map<std::thread::id, MemoryBuffer*> TreadMemory;
  typedef std::map<std::thread::id, BufferName*> TreadName;

  static char* SharedEnv() {
    static char* shared_env = getenv("SHARED_INST_NUM");
    return shared_env;
  }
  static const int SharedInstNum() {
    static const int shared_inst_num =
        (getenv("SHARED_INST_NUM") != nullptr) ? std::atoi(getenv("SHARED_INST_NUM")) : 1;
    return shared_inst_num;
  }

  static MemoryBuffer& Buffer() {
    static TreadMemory t_memory;
    // (TODO) it's not good for each thread to obtain a MemoryBuffer
    std::thread::id id = std::this_thread::get_id();
    if (t_memory.count(id) == 0) {
      t_memory[id] = new MemoryBuffer();
    }
    return *(t_memory[id]);
  }

  static BufferName& Name() {
    static TreadName t_name;
    // (TODO) it's not good for each thread to obtain a MemoryBuffer
    std::thread::id id = std::this_thread::get_id();
    if (t_name.count(id) == 0) {
      t_name[id] = new BufferName();
    }
    return *(t_name[id]);
  }

  static void SetName(void* data, const string name) {
    BufferName& name_buffer = Name();
    MemoryBuffer& memory_buffer = Buffer();
    auto iter = memory_buffer.find(data);
    if (iter != memory_buffer.end()) {
      name_buffer[iter->first] = name;
    } else {
      LOG(WARNING) << "name a not existing pointer...";
    }
  }

  static int AliveBuffer() {
    MemoryBuffer& memory_buffer = Buffer();
    BufferName& name_buffer = Name();
    int alive = 0;
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      if (buffer_count != 0) {
        alive++;
        LOG(WARNING) << "have alive buffer name " << name_buffer[iter->first];
      }
    }
    LOG(WARNING) << "buffer alive count " << alive;
    return alive;
  }

  static void ReleaseBuffer() {
    MemoryBuffer& memory_buffer = Buffer();
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      if (buffer_count != 0) {
        LOG(WARNING) << "buffer still have life, force release...";
        iter->second[0] = 0;
      }
    }
  }

  static StrategyList& Strategy() {
    static StrategyList* m_strategy_ =
        new StrategyList({{"cycle_buffer", false}, {"direct_buffer", false}, {"unified_buffer", false}});
    return *m_strategy_;
  }

  static ipc::managed_shared_memory& ManagedShm(char* space_name = "SharedWeight") {
    static ipc::managed_shared_memory shm_ptr(ipc::open_only, space_name);
    return shm_ptr;
  }

  static void InitStrategy() {
    string memory_strategy = getenv("UNIFIED_BUFFER") != NULL
                                 ? "unified_buffer"
                                 : (getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer");
    SetStrategy(memory_strategy);
  }

  static void SetStrategy(const string strategy) {
    CHECK(strategy == "cycle_buffer" || strategy == "direct_buffer" || strategy == "unified_buffer")
        << "only support memory strategy cycle buffer, direct buffer and unified buffer";
    StrategyList& strategy_list = Strategy();
    strategy_list[strategy] = true;
    LOG(INFO) << "strategy list set success " << strategy;
  }

  static int CheckMemory(void* data) {
    MemoryBuffer& memory_buffer = Buffer();
    auto iter = memory_buffer.find(data);
    if (iter != memory_buffer.end()) {
      return iter->second[0];
    } else {
      LOG(WARNING) << "get life from a not existing memory pointer...";
      return -1;
    }
  }

  // set the data buffer a new life count
  static void ResetMemory(void* data, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    StrategyList& strategy_list = Strategy();
    auto iter = memory_buffer.find(data);
    if (iter != memory_buffer.end()) {
      iter->second[0] = life_count;
    } else {
      LOG(WARNING) << "reset a not existing memory pointer...";
    }
  }

  // will return the left count of one tensor
  static int UnrefMemory(void* data, bool inplace = false) {
    MemoryBuffer& memory_buffer = Buffer();
    StrategyList& strategy_list = Strategy();
    auto iter = memory_buffer.find(data);
    int status = 0;
    if (iter != memory_buffer.end()) {
      if (iter->second[0] <= 0) {
        LOG(WARNING) << "free a no-used memory...";
        iter->second[0] = 0;
        status = 0;
      } else {
        (iter->second[0])--;
        status = iter->second[0];
      }
      if (status == 0 && inplace == false) {
        if (strategy_list["direct_buffer"]) {
          auto free_ptr = iter->first;
          free(free_ptr);
          memory_buffer.erase(free_ptr);
        } else if (strategy_list["unified_buffer"]) {
          auto free_ptr = iter->first;
          i_free(free_ptr);
          memory_buffer.erase(free_ptr);
        }
      }
    } else {
      LOG(WARNING) << "free not existing memory pointer...";
      status = -1;
    }
    return status;
  }

  static void* GetMemory(size_t size, const int life_count) {
    if (size == 0) {
      LOG(INFO) << "please set the tensor size...";
      return nullptr;
    }
    if (life_count <= 0) {
      LOG(INFO) << "please set the tensor life...";
      return nullptr;
    }
    StrategyList& strategy_list = Strategy();
    if (strategy_list["direct_buffer"]) {
      return DirectBufferGetMemory(size, life_count);
    } else if (strategy_list["cycle_buffer"]) {
      return CycleBufferGetMemory(size, life_count);
    } else if (strategy_list["unified_buffer"]) {
      return UnifiedBufferGetMemory(size, life_count);
    } else {
      LOG(ERROR) << "please set the memory strategy";
      return nullptr;
    }
  }

  static void* CycleBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    LOG(INFO) << "cycle buffer tensor size is " << memory_buffer.size();
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      auto buffer_size = iter->second[1];
      if (buffer_count == 0) {
        if (size > buffer_size) {
          auto free_ptr = iter->first;
          free(free_ptr);
          memory_buffer.erase(free_ptr);
          // allocate new buffer
          void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
          memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
          return buf;
        } else {
          memory_buffer[iter->first] = vector<size_t>({static_cast<size_t>(life_count), buffer_size});
          return iter->first;
        }
      }
    }
    // allocate new buffer
    void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  static void* DirectBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    LOG(INFO) << "direct buffer tensor size is " << memory_buffer.size();
    void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  static void* UnifiedBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    LOG(INFO) << "unified buffer tensor size is " << memory_buffer.size();
    void* buf = reinterpret_cast<void*>(i_malloc(size));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  // MemoryAllocator(MemoryAllocator const&)  = delete;
  // void operator=(MemoryAllocator const&)  = delete;

  static MemoryAllocator& get() {
    static MemoryAllocator instance;
    return instance;
  }

 private:
  // Private constructor to prevent instancing.
  MemoryAllocator() {}
};

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_
