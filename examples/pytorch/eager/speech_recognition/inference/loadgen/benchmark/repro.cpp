/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

class QSL : public mlperf::QuerySampleLibrary {
 public:
  ~QSL() override{};
  const std::string& Name() const override { return mName; }
  size_t TotalSampleCount() override { return 1000000; }
  size_t PerformanceSampleCount() override { return TotalSampleCount(); }
  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {}
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {}

 private:
  std::string mName{"Dummy QSL"};
};

class BasicSUT : public mlperf::SystemUnderTest {
 public:
  BasicSUT() {
    // Start with some large value so that we don't reallocate memory.
    initResponse(10000);
  }
  ~BasicSUT() override {}
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    int n = samples.size();
    if (n > mResponses.size()) {
      std::cerr << "Warning: reallocating response buffer in BasicSUT. Maybe "
                   "you should initResponse with larger value!?"
                << std::endl;
      initResponse(samples.size());
    }
    for (int i = 0; i < n; i++) {
      mResponses[i].id = samples[i].id;
    }
    mlperf::QuerySamplesComplete(mResponses.data(), n);
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void initResponse(int size) {
    mResponses.resize(size,
                      {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
  }
  int mBuf{0};
  std::string mName{"BasicSUT"};
  std::vector<mlperf::QuerySampleResponse> mResponses;
};

class QueueSUT : public mlperf::SystemUnderTest {
 public:
  QueueSUT(int numCompleteThreads, int maxSize) {
    // Each thread handle at most maxSize at a time.
    std::cout << "QueueSUT: maxSize = " << maxSize << std::endl;
    initResponse(numCompleteThreads, maxSize);
    // Launch complete threads
    for (int i = 0; i < numCompleteThreads; i++) {
      mThreads.emplace_back(&QueueSUT::CompleteThread, this, i);
    }
  }
  ~QueueSUT() override {
    {
      std::unique_lock<std::mutex> lck(mMtx);
      mDone = true;
      mCondVar.notify_all();
    }
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::unique_lock<std::mutex> lck(mMtx);
    for (const auto& sample : samples) {
      mIdQueue.push_back(sample.id);
    }
    // Let some worker thread to consume tasks
    mCondVar.notify_one();
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void CompleteThread(int threadIdx) {
    auto& responses = mResponses[threadIdx];
    size_t maxSize{responses.size()};
    size_t actualSize{0};
    while (true) {
      {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return !mIdQueue.empty() || mDone; });

        if (mDone) {
          break;
        }

        actualSize = std::min(maxSize, mIdQueue.size());
        for (int i = 0; i < actualSize; i++) {
          responses[i].id = mIdQueue.front();
          mIdQueue.pop_front();
        }
        mCondVar.notify_one();
      }
      mlperf::QuerySamplesComplete(responses.data(), actualSize);
    }
  }
  void initResponse(int numCompleteThreads, int size) {
    mResponses.resize(numCompleteThreads);
    for (auto& responses : mResponses) {
      responses.resize(size,
                       {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
    }
  }
  int mBuf{0};
  std::string mName{"QueueSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::vector<std::thread> mThreads;
  std::deque<mlperf::ResponseId> mIdQueue;
  std::mutex mMtx;
  std::condition_variable mCondVar;
  bool mDone{false};
};

class MultiBasicSUT : public mlperf::SystemUnderTest {
 public:
  MultiBasicSUT(int numThreads)
      : mNumThreads(numThreads), mResponses(numThreads) {
    // Start with some large value so that we don't reallocate memory.
    initResponse(10000);
    for (int i = 0; i < mNumThreads; ++i) {
      mThreads.emplace_back(&MultiBasicSUT::startIssueThread, this, i);
    }
  }
  ~MultiBasicSUT() override {
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    int thread_idx = mThreadMap[std::this_thread::get_id()];
    int n = samples.size();
    auto& reponses = mResponses[thread_idx];
    if (n > reponses.size()) {
      std::cout
          << "Warning: reallocating response buffer in MultiBasicSUT. Maybe "
             "you should initResponse with larger value!?"
          << std::endl;
      initResponse(samples.size());
    }
    for (int i = 0; i < n; i++) {
      reponses[i].id = samples[i].id;
    }
    mlperf::QuerySamplesComplete(reponses.data(), n);
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void initResponse(int size) {
    for (auto& responses : mResponses) {
      responses.resize(size,
                       {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
    }
  }
  void startIssueThread(int thread_idx) {
    {
      std::lock_guard<std::mutex> lock(mMtx);
      mThreadMap[std::this_thread::get_id()] = thread_idx;
    }
    mlperf::RegisterIssueQueryThread();
  }
  int mBuf{0};
  int mNumThreads{0};
  std::string mName{"MultiBasicSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::mutex mMtx;
  std::vector<std::thread> mThreads;
  std::map<std::thread::id, int> mThreadMap;
};

int main(int argc, char** argv) {
  assert(argc >= 2 && "Need to pass in at least one argument: target_qps");
  int target_qps = std::stoi(argv[1]);
  std::cout << "target_qps = " << target_qps << std::endl;

  bool useQueue{false};
  int numCompleteThreads{4};
  int maxSize{1};
  bool server_coalesce_queries{false};
  int num_issue_threads{0};
  if (argc >= 3) {
    useQueue = std::stoi(argv[2]) != 0;
  }
  if (argc >= 4) {
    numCompleteThreads = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    maxSize = std::stoi(argv[4]);
  }
  if (argc >= 6) {
    server_coalesce_queries = std::stoi(argv[5]) != 0;
  }
  if (argc >= 7) {
    num_issue_threads = std::stoi(argv[6]);
  }

  QSL qsl;
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  mlperf::TestSettings testSettings;
  testSettings.scenario = mlperf::TestScenario::Server;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.server_target_qps = target_qps;
  testSettings.server_target_latency_ns = 10000000;  // 10ms
  testSettings.server_target_latency_percentile = 0.99;
  testSettings.min_duration_ms = 60000;
  testSettings.min_query_count = 270000;
  testSettings.server_coalesce_queries = server_coalesce_queries;
  std::cout << "testSettings.server_coalesce_queries = "
            << (server_coalesce_queries ? "True" : "False") << std::endl;
  testSettings.server_num_issue_query_threads = num_issue_threads;
  std::cout << "num_issue_threads = " << num_issue_threads << std::endl;

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = "build";
  logSettings.log_output.prefix = "mlperf_log_";
  logSettings.log_output.suffix = "";
  logSettings.log_output.prefix_with_datetime = false;
  logSettings.log_output.copy_detail_to_stdout = false;
  logSettings.log_output.copy_summary_to_stdout = true;
  logSettings.log_mode = mlperf::LoggingMode::AsyncPoll;
  logSettings.log_mode_async_poll_interval_ms = 1000;
  logSettings.enable_trace = false;

  // Choose SUT
  if (num_issue_threads == 0) {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << numCompleteThreads
                << " complete threads" << std::endl;
      sut.reset(new QueueSUT(numCompleteThreads, maxSize));
    } else {
      std::cout << "Using BasicSUT" << std::endl;
      sut.reset(new BasicSUT());
    }
  } else {
    if (useQueue) {
      std::cout << "Using MultiQueueSUT with " << numCompleteThreads
                << " complete threads" << std::endl;
      std::cerr << "!!!! MultiQueueSUT is NOT implemented yet !!!!"
                << std::endl;
      return 1;
      // sut.reset(new MultiQueueSUT(num_issue_threads, numCompleteThreads,
      // maxSize));
    } else {
      std::cout << "Using MultiBasicSUT" << std::endl;
      sut.reset(new MultiBasicSUT(num_issue_threads));
    }
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  return 0;
}
