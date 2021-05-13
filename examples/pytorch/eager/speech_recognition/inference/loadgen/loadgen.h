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
/// \brief Provides the entry points for a SUT to start a test and respond
/// to issued queries.

#ifndef MLPERF_LOADGEN_LOADGEN_H_
#define MLPERF_LOADGEN_LOADGEN_H_

#include <cstddef>

/// \brief Contains the loadgen API.
namespace mlperf {

struct QuerySampleResponse;
class QuerySampleLibrary;
class SystemUnderTest;
struct TestSettings;
struct LogSettings;

/// \addtogroup LoadgenAPI Loadgen API
/// @{

///
/// \brief SUT calls this to notify loadgen of completed samples.
/// \details
/// * The samples may be from any combination of queries or partial queries as
///   issued by \link mlperf::SystemUnderTest::IssueQuery
///   SystemUnderTest::IssueQuery \endlink.
/// * The SUT is responsible for allocating and owning the response data
///   which must remain valid for the duration of this call. The loadgen
///   will copy the response data if needed (e.g. for accuracy mode).
///   + Note: This setup requires the allocation to be timed, which
///     will benefit SUTs that efficiently recycle response memory.
/// * All calls to QuerySampleComplete are thread-safe and wait-free bounded.
///   + Any number of threads can call QuerySampleComplete simultaneously.
///   + Regardless of where any other thread stalls, the current thread will
///     finish QuerySampleComplete in a bounded number of cycles.
void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count);

///
/// \brief Starts the test against SUT with the specified settings.
/// \details This is the C++ entry point. See mlperf::c::StartTest for the
/// C entry point.
///
void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings);

///
/// \brief Aborts the running test.
/// \details This function will stop issueing new samples to the SUT. StartTest
/// will return after the current inference finishes. Since StartTest is a
/// blocking function, this function can only be called in another thread.
void AbortTest();

///
/// \brief Register a thread for query issuing in Server scenario.
/// \details If a thread registers itself, the thread(s) is used to call SUT's
/// IssueQuery(). This function is blocking until the entire test is done. The
/// number of registered threads must match server_num_issue_query_threads in
/// TestSettings. This function only has effect in Server scenario.
/// This is the C++ entry point. See mlperf::c::RegisterIssueQueryThread for the
/// C entry point.
///
void RegisterIssueQueryThread();

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_LOADGEN_H_
