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

#ifndef ENGINE_EXECUTOR_INCLUDE_THREAD_POOL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_THREAD_POOL_HPP_

#include <pthread.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <utility>

/**
 * @brief A thread pool for Multi Stream parallel execution.
 */
class ThreadPool {
 private:
  using Task = std::function<void()>;
  // thread container
  std::vector<std::thread> pool;
  // is each thread stoped
  std::vector<bool> stoped;
  // queue of tasks
  std::queue<Task> tasks;
  // sync of multi stream
  std::mutex tasks_lock;
  std::condition_variable task_cond_var;
  std::atomic<unsigned int> idle_thread_num;
  std::atomic<unsigned int> work_thread_num;
  // is thread pool stoped
  bool pool_stoped;

 private:
  void _initPool_() {
    unsigned int index = pool.size();
    stoped.emplace_back(false);
    idle_thread_num++;
    pool.emplace_back([this, index] {
      while (true) {
        // capature the task
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->tasks_lock);
          this->task_cond_var.wait(lock, [this, index] {
            return this->stoped[index] || !this->tasks.empty();
          });  // wait utill capature the task
          if (this->stoped[index]) {
            idle_thread_num--;
            return;
          }
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        {
          idle_thread_num--, work_thread_num++;
          task();  // run the task
          idle_thread_num++, work_thread_num--;
        }
      }
    });
  }

 public:
  inline ThreadPool() {
    work_thread_num = 0;
    pool_stoped = false;
  }
  // wait for all threads to finish and stop all threads
  inline ~ThreadPool() {
    for (auto item : stoped) item = true;
    task_cond_var.notify_all();  // wake up all thread to run
    for (auto& th : pool) {
      if (th.joinable()) th.join();
    }
  }

 public:
  /**
   * @brief commit a task to thread pool and assign a worker from queue to run
   * this task.
   * @param func: the function of the task could use std::bind for class method
   * @param args: zero or more arguments for the func
   *        example: .commitTask(std::bind(&ZCQ::helloWorld, &zcq));
   * @return A future to be used later to check if the function has finished its
   * execution.
   */
  template <typename FUNC, typename... Args>
  auto commitTask(FUNC&& func, Args&&... args)
      -> std::future<decltype(func(args...))> {
    if (hasStopedPool()) throw std::runtime_error("the threadPool is stopped.");
    using ReturnType = decltype(func(args...));
    auto task = std::make_shared<std::packaged_task<ReturnType()> >(
        std::bind(std::forward<FUNC>(func), std::forward<Args>(args)...));
    std::future<ReturnType> future = task->get_future();
    {  // add to task queue
      std::lock_guard<std::mutex> lock(tasks_lock);
      tasks.emplace([task]() { (*task)(); });
    }
    task_cond_var.notify_one();  // wake up on worker to get run task
    return future;
  }

  // init the thread pool with size;
  void begin(unsigned int size) {
    pool_stoped = false;
    for (unsigned int s = 0; s < size; s++) _initPool_();
  }

  // adjust the size of pool
  void resize(unsigned int sz) {
    pool_stoped = false;
    // alive thread num
    size_t as = idle_thread_num + work_thread_num;
    // present pool size
    size_t ps = pool.size(), rs = 0;
    if (sz > as) {
      for (unsigned int s = as; s < sz; s++) _initPool_();
    }
    if (sz < as) {
      for (auto s : stoped) {
        if (!s) { s = true; task_cond_var.notify_all(); rs++; }
        if (rs == as - sz) break;
      }
    }
  }

  // idle thread num
  unsigned int idleNum() { return idle_thread_num; }
  // work thread num
  unsigned int workNum() { return work_thread_num; }
  // stop task commit
  void stopTask() { pool_stoped = true; }
  // restart task
  void restartTask() { pool_stoped = false; }
  // close thread pool and release resource
  void close() {
    for (auto a : stoped) a = true;
    task_cond_var.notify_all();
    pool_stoped = true;
  }
  bool hasStopedPool() { return pool_stoped; }
  // wait for all tasks to be completed
  void waitAllTaskRunOver() {
    while (true) {
      if (work_thread_num == 0) {
        return;
      } else {
        std::this_thread::yield();
      }
    }
  }
};

#endif  // ENGINE_EXECUTOR_INCLUDE_THREAD_POOL_HPP_
