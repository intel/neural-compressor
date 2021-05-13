/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "c_api.h"

#include <string>

#include "../loadgen.h"
#include "../query_sample.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"

namespace mlperf {
namespace c {
namespace {

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(
      ClientData client_data, std::string name, IssueQueryCallback issue_cb,
      FlushQueriesCallback flush_queries_cb,
      ReportLatencyResultsCallback report_latency_results_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        issue_cb_(issue_cb),
        flush_queries_cb_(flush_queries_cb),
        report_latency_results_cb_(report_latency_results_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    (*issue_cb_)(client_data_, samples.data(), samples.size());
  }

  void FlushQueries() override { (*flush_queries_cb_)(); }

  void ReportLatencyResults(
      const std::vector<QuerySampleLatency>& latencies_ns) override {
    (*report_latency_results_cb_)(client_data_, latencies_ns.data(),
                                  latencies_ns.size());
  }

 private:
  ClientData client_data_;
  std::string name_;
  IssueQueryCallback issue_cb_;
  FlushQueriesCallback flush_queries_cb_;
  ReportLatencyResultsCallback report_latency_results_cb_;
};

}  // namespace

void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb,
                   FlushQueriesCallback flush_queries_cb,
                   ReportLatencyResultsCallback report_latency_results_cb) {
  SystemUnderTestTrampoline* sut = new SystemUnderTestTrampoline(
      client_data, std::string(name, name_length), issue_cb, flush_queries_cb,
      report_latency_results_cb);
  return reinterpret_cast<void*>(sut);
}

void DestroySUT(void* sut) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  delete sut_cast;
}

namespace {

// Forwards QuerySampleLibrary calls to relevant callbacks.
class QuerySampleLibraryTrampoline : public QuerySampleLibrary {
 public:
  QuerySampleLibraryTrampoline(
      ClientData client_data, std::string name, size_t total_sample_count,
      size_t performance_sample_count,
      LoadSamplesToRamCallback load_samples_to_ram_cb,
      UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb)
      : client_data_(client_data),
        name_(std::move(name)),
        total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count),
        load_samples_to_ram_cb_(load_samples_to_ram_cb),
        unload_samlpes_from_ram_cb_(unload_samlpes_from_ram_cb) {}
  ~QuerySampleLibraryTrampoline() override = default;

  const std::string& Name() const override { return name_; }
  size_t TotalSampleCount() override { return total_sample_count_; }
  size_t PerformanceSampleCount() override { return performance_sample_count_; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    (*load_samples_to_ram_cb_)(client_data_, samples.data(), samples.size());
  }
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {
    (*unload_samlpes_from_ram_cb_)(client_data_, samples.data(),
                                   samples.size());
  }

 private:
  ClientData client_data_;
  std::string name_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  LoadSamplesToRamCallback load_samples_to_ram_cb_;
  UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb_;
};

}  // namespace

void* ConstructQSL(ClientData client_data, const char* name, size_t name_length,
                   size_t total_sample_count, size_t performance_sample_count,
                   LoadSamplesToRamCallback load_samples_to_ram_cb,
                   UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb) {
  QuerySampleLibraryTrampoline* qsl = new QuerySampleLibraryTrampoline(
      client_data, std::string(name, name_length), total_sample_count,
      performance_sample_count, load_samples_to_ram_cb,
      unload_samlpes_from_ram_cb);
  return reinterpret_cast<void*>(qsl);
}

void DestroyQSL(void* qsl) {
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  delete qsl_cast;
}

// mlperf::c::StartTest just forwards to mlperf::StartTest after doing the
// proper cast.
void StartTest(void* sut, void* qsl, const TestSettings& settings) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  LogSettings default_log_settings;
  mlperf::StartTest(sut_cast, qsl_cast, settings, default_log_settings);
}

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  mlperf::QuerySamplesComplete(responses, response_count);
}

void RegisterIssueQueryThread() { mlperf::RegisterIssueQueryThread(); }

}  // namespace c
}  // namespace mlperf
