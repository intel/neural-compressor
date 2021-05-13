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

/// \file
/// \brief A C API wrapping the C++ loadgen. Not tested. Needs work.
/// \details The C API allows a C or Python client to easily create
/// a SystemUnderTest without having to expose the SystemUnderTest class
/// directly.
/// ConstructSUT works with a bunch of function poitners instead that are
/// called from an underlying trampoline class.

#ifndef SYSTEM_UNDER_TEST_C_API_H_
#define SYSTEM_UNDER_TEST_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "../query_sample.h"
#include "../test_settings.h"

namespace mlperf {

namespace c {

/// \brief Optional opaque client data that creators of SUTs and QSLs can have
/// the loadgen pass back to their callback invocations.
/// Helps avoids global variables.
typedef uintptr_t ClientData;

typedef void (*IssueQueryCallback)(ClientData, const QuerySample*, size_t);
typedef void (*FlushQueriesCallback)();
typedef void (*ReportLatencyResultsCallback)(ClientData, const int64_t*,
                                             size_t);

/// \brief SUT calls this function to report query result back to loadgen
void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count);

/// \brief Create an opaque SUT pointer based on C callbacks.
void* ConstructSUT(ClientData client_data, const char* name, size_t name_length,
                   IssueQueryCallback issue_cb,
                   FlushQueriesCallback flush_queries_cb,
                   ReportLatencyResultsCallback report_latency_results_cb);
/// \brief Destroys the SUT created by ConstructSUT.
void DestroySUT(void* sut);

typedef void (*LoadSamplesToRamCallback)(ClientData, const QuerySampleIndex*,
                                         size_t);
typedef void (*UnloadSamplesFromRamCallback)(ClientData,
                                             const QuerySampleIndex*, size_t);

/// \brief Create an opaque QSL pointer based on C callbacks.
void* ConstructQSL(ClientData client_data, const char* name, size_t name_length,
                   size_t total_sample_count, size_t performance_sample_count,
                   LoadSamplesToRamCallback load_samples_to_ram_cb,
                   UnloadSamplesFromRamCallback unload_samlpes_from_ram_cb);
/// \brief Destroys the QSL created by ConsructQSL.
void DestroyQSL(void* qsl);

/// \brief Run tests on a SUT created by ConstructSUT().
/// \details This is the C entry point. See mlperf::StartTest for the C++ entry
/// point.
void StartTest(void* sut, void* qsl, const TestSettings& settings);

///
/// \brief Register a thread for query issuing in Server scenario.
/// \details This is the C entry point. See mlperf::RegisterIssueQueryThread for the C++ entry
/// point.
///
void RegisterIssueQueryThread();

}  // namespace c
}  // namespace mlperf

#endif  // SYSTEM_UNDER_TEST_C_API_H_
