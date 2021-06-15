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
/// \brief Implements IssueQueryController and other helper classes for
/// query issuing.

#include "issue_query_controller.h"

#include <sstream>

namespace mlperf {

void RegisterIssueQueryThread() {
  loadgen::IssueQueryController::GetInstance().RegisterThread();
}

/// \brief Loadgen implementation details.
namespace loadgen {

QueryMetadata::QueryMetadata(
    const std::vector<QuerySampleIndex>& query_sample_indices,
    std::chrono::nanoseconds scheduled_delta,
    ResponseDelegate* response_delegate, SequenceGen* sequence_gen)
    : scheduled_delta(scheduled_delta),
      response_delegate(response_delegate),
      sequence_id(sequence_gen->NextQueryId()),
      wait_count_(query_sample_indices.size()) {
  samples_.reserve(query_sample_indices.size());
  for (QuerySampleIndex qsi : query_sample_indices) {
    samples_.push_back({this, sequence_gen->NextSampleId(), qsi,
                        sequence_gen->NextAccLogRng()});
  }
  query_to_send.reserve(query_sample_indices.size());
  for (auto& s : samples_) {
    query_to_send.push_back({reinterpret_cast<ResponseId>(&s), s.sample_index});
  }
}

QueryMetadata::QueryMetadata(QueryMetadata&& src)
    : query_to_send(std::move(src.query_to_send)),
      scheduled_delta(src.scheduled_delta),
      response_delegate(src.response_delegate),
      sequence_id(src.sequence_id),
      wait_count_(src.samples_.size()),
      samples_(std::move(src.samples_)) {
  // The move constructor should only be called while generating a
  // vector of QueryMetadata, before it's been used.
  // Assert that wait_count_ is in its initial state.
  assert(src.wait_count_.load() == samples_.size());
  // Update the "parent" of each sample to be this query; the old query
  // address will no longer be valid.
  // TODO: Only set up the sample parenting once after all the queries have
  //       been created, rather than re-parenting on move here.
  for (size_t i = 0; i < samples_.size(); i++) {
    SampleMetadata* s = &samples_[i];
    s->query_metadata = this;
    query_to_send[i].id = reinterpret_cast<ResponseId>(s);
  }
}

void QueryMetadata::NotifyOneSampleCompleted(PerfClock::time_point timestamp) {
  size_t old_count = wait_count_.fetch_sub(1, std::memory_order_relaxed);
  if (old_count == 1) {
    all_samples_done_time = timestamp;
    all_samples_done_.set_value();
    response_delegate->QueryComplete();
  }
}

void QueryMetadata::WaitForAllSamplesCompleted() {
  all_samples_done_.get_future().wait();
}

PerfClock::time_point QueryMetadata::WaitForAllSamplesCompletedWithTimestamp() {
  all_samples_done_.get_future().wait();
  return all_samples_done_time;
}

// When server_coalesce_queries is set to true in Server scenario, we
// sometimes coalesce multiple queries into one query. This is done by moving
// the other query's sample into current query, while maintaining their
// original scheduled_time.
void QueryMetadata::CoalesceQueries(QueryMetadata* queries, size_t first,
                                    size_t last, size_t stride) {
  // Copy sample data over to current query, boldly assuming that each query
  // only has one sample.
  query_to_send.reserve((last - first) / stride +
                        2);  // Extra one for the current query.
  for (size_t i = first; i <= last; i += stride) {
    auto& q = queries[i];
    auto& s = q.samples_[0];
    query_to_send.push_back({reinterpret_cast<ResponseId>(&s), s.sample_index});
    q.scheduled_time = scheduled_time + q.scheduled_delta - scheduled_delta;
    q.issued_start_time = issued_start_time;
  }
}

void QueryMetadata::Decoalesce() { query_to_send.resize(1); }

/// \brief A base template that should never be used since each scenario has
/// its own specialization.
template <TestScenario scenario>
struct QueryScheduler {
  static_assert(scenario != scenario, "Unhandled TestScenario");
};

/// \brief Schedules queries for issuance in the single stream scenario.
template <>
struct QueryScheduler<TestScenario::SingleStream> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer = MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
    if (prev_query != nullptr) {
      prev_query->WaitForAllSamplesCompleted();
    }
    prev_query = next_query;

    auto now = PerfClock::now();
    next_query->scheduled_time = now;
    next_query->issued_start_time = now;
    return now;
  }

  QueryMetadata* prev_query = nullptr;
};

/// \brief Schedules queries for issuance in the multi stream scenario.
template <>
struct QueryScheduler<TestScenario::MultiStream> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point start)
      : qps(settings.target_qps),
        max_async_queries(settings.max_async_queries),
        start_time(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        prev_queries.front()->WaitForAllSamplesCompleted();
        prev_queries.pop();
      }
    }

    {
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });
      // TODO(brianderson): Skip ticks based on the query complete time,
      //     before the query synchronization + notification thread hop,
      //     rather than after.
      PerfClock::time_point now = PerfClock::now();
      auto i_period_old = i_period;
      PerfClock::time_point tick_time;
      do {
        i_period++;
        tick_time =
            start_time + SecondsToDuration<PerfClock::duration>(i_period / qps);
        Log([tick_time](AsyncLog& log) {
          log.TraceAsyncInstant("QueryInterval", 0, tick_time);
        });
      } while (tick_time < now);
      next_query->scheduled_intervals = i_period - i_period_old;
      next_query->scheduled_time = tick_time;
      std::this_thread::sleep_until(tick_time);
    }

    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  size_t i_period = 0;
  double qps;
  const size_t max_async_queries;
  PerfClock::time_point start_time;
  std::queue<QueryMetadata*> prev_queries;
};

/// \brief Schedules queries for issuance in the single stream free scenario.
template <>
struct QueryScheduler<TestScenario::MultiStreamFree> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point /*start*/)
      : max_async_queries(settings.max_async_queries) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    bool schedule_time_needed = true;
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        next_query->scheduled_time =
            prev_queries.front()->WaitForAllSamplesCompletedWithTimestamp();
        schedule_time_needed = false;
        prev_queries.pop();
      }
    }

    auto now = PerfClock::now();
    if (schedule_time_needed) {
      next_query->scheduled_time = now;
    }
    next_query->issued_start_time = now;
    return now;
  }

  const size_t max_async_queries;
  std::queue<QueryMetadata*> prev_queries;
};

/// \brief Schedules queries for issuance in the server scenario.
template <>
struct QueryScheduler<TestScenario::Server> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer =
        MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });

    auto scheduled_time = start + next_query->scheduled_delta;
    next_query->scheduled_time = scheduled_time;

    auto now = PerfClock::now();
    if (now < scheduled_time) {
      std::this_thread::sleep_until(scheduled_time);
      now = PerfClock::now();
    }
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

/// \brief Schedules queries for issuance in the offline scenario.
template <>
struct QueryScheduler<TestScenario::Offline> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    next_query->scheduled_time = start;
    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

IssueQueryController& IssueQueryController::GetInstance() {
  // The singleton.
  static IssueQueryController instance;
  return instance;
}

void IssueQueryController::RegisterThread() {
  // Push this thread to thread queue.
  auto thread_id = std::this_thread::get_id();
  size_t thread_idx{0};
  {
    std::lock_guard<std::mutex> lock(mtx);
    thread_idx = thread_ids.size();
    thread_ids.emplace_back(thread_id);
  }

  LogDetail([thread_id, thread_idx](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    std::stringstream ss;
    ss << "Registered IssueQueryThread[" << thread_idx
       << "]. thread ID : " << std::hash<std::thread::id>()(thread_id);
    MLPERF_LOG(detail, "generic_message", ss.str());
#else
    detail("Registered IssueQueryThread[" + std::to_string(thread_idx) +
               "]. thread ID : ",
           std::to_string(std::hash<std::thread::id>()(thread_id)));
#endif
  });

  // Start test.
  while (true) {
    // Wait until the main thread signals a start or the end.
    {
      std::unique_lock<std::mutex> lock(mtx);
      cond_var.wait(lock, [this]() { return issuing || end_test; });
      // The test has ended.
      if (end_test) {
        break;
      }
    }

    // Start issuing queries.
    if (thread_idx <= num_threads) {
      IssueQueriesInternal<TestScenario::Server, true>(num_threads, thread_idx);
      {
        std::lock_guard<std::mutex> lock(mtx);
        thread_complete[thread_idx] = true;
      }
      cond_var.notify_all();
    }

    // Wait until all issue threads complete.
    {
      std::unique_lock<std::mutex> lock(mtx);
      cond_var.wait(lock, [this]() { return !issuing; });
    }
  }
}

void IssueQueryController::SetNumThreads(size_t n) {
  // Try waiting for IssueQueryThreads() to registered themselves.
  std::unique_lock<std::mutex> lock(mtx);
  const std::chrono::seconds timeout(10);
  num_threads = n;
  cond_var.wait_for(lock, timeout,
                    [this]() { return thread_ids.size() >= num_threads; });
  // If the number of registered threads do not match the settings, report an
  // error.
  if (num_threads != thread_ids.size()) {
    LogDetail([this](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "Mismatch between settings and number of registered "
         << "IssueQueryThreads! settings.server_num_issue_query_threads = "
         << num_threads << " but " << thread_ids.size()
         << " threads registered.";
      MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
      detail.Error(
          "Mismatch between settings and number of registered ",
          "IssueQueryThreads! settings.server_num_issue_query_threads = ",
          num_threads, " but ", thread_ids.size(), " threads registered.");
#endif
    });
  }
}

template <TestScenario scenario>
void IssueQueryController::StartIssueQueries(IssueQueryState* s) {
  // Get the state.
  state = s;
  state->start_for_power = std::chrono::system_clock::now();
  state->start_time = PerfClock::now();

  if (scenario != TestScenario::Server || num_threads == 0) {
    // Usually, we just use the same thread to issue queries.
    IssueQueriesInternal<scenario, false>(1, 0);
  } else {
    // If server_num_issue_query_threads is non-zero, issue queries on the
    // registered threads.
    // Tell all threads to start issuing queries.
    {
      std::unique_lock<std::mutex> lock(mtx);
      issuing = true;
      thread_complete.assign(num_threads, false);
    }
    cond_var.notify_all();
    // Wait until all issue threads complete.
    {
      std::unique_lock<std::mutex> lock(mtx);
      cond_var.wait(lock, [this]() {
        return std::all_of(thread_complete.begin(), thread_complete.end(),
                           [](bool in) { return in; });
      });
      issuing = false;
    }
    cond_var.notify_all();
  }
}

template void IssueQueryController::StartIssueQueries<
    TestScenario::MultiStream>(IssueQueryState* s);
template void IssueQueryController::StartIssueQueries<
    TestScenario::MultiStreamFree>(IssueQueryState* s);
template void IssueQueryController::StartIssueQueries<TestScenario::Offline>(
    IssueQueryState* s);
template void IssueQueryController::StartIssueQueries<TestScenario::Server>(
    IssueQueryState* s);
template void IssueQueryController::StartIssueQueries<
    TestScenario::SingleStream>(IssueQueryState* s);

void IssueQueryController::EndThreads() {
  // Tell all the issue threads to end.
  {
    std::lock_guard<std::mutex> lock(mtx);
    end_test = true;
  }
  cond_var.notify_all();
}

template <TestScenario scenario, bool multi_thread>
void IssueQueryController::IssueQueriesInternal(size_t query_stride,
                                                size_t thread_idx) {
  // Get all the needed information.
  auto sut = state->sut;
  auto& queries = *state->queries;
  auto& response_logger = *state->response_delegate;

  // Some book-keeping about the number of queries issued.
  size_t queries_issued = 0;
  size_t queries_issued_per_iter = 0;
  size_t queries_count = queries.size();

  // Calculate the min/max queries per issue thread.
  const auto& settings = *state->settings;
  const size_t min_query_count = settings.min_query_count;
  const size_t min_query_count_for_thread =
      (thread_idx < (min_query_count % query_stride))
          ? (min_query_count / query_stride + 1)
          : (min_query_count / query_stride);
  const size_t max_query_count = settings.max_query_count;
  const size_t max_query_count_for_thread =
      (thread_idx < (max_query_count % query_stride))
          ? (max_query_count / query_stride + 1)
          : (max_query_count / query_stride);

  // Create query scheduler.
  const auto start = state->start_time;
  QueryScheduler<scenario> query_scheduler(settings, start);
  auto last_now = start;

  // We can never run out of generated queries in the server scenario,
  // since the duration depends on the scheduled query time and not
  // the actual issue time.
  bool ran_out_of_generated_queries = scenario != TestScenario::Server;
  // This is equal to the sum of numbers of samples issued.
  size_t expected_latencies = 0;

  for (size_t queries_idx = thread_idx; queries_idx < queries_count;
       queries_idx += query_stride) {
    queries_issued_per_iter = 0;
    auto& query = queries[queries_idx];
    auto tracer1 =
        MakeScopedTracer([](AsyncTrace& trace) { trace("SampleLoop"); });
    last_now = query_scheduler.Wait(&query);

    // If in Server scenario and server_coalesce_queries is enabled, multiple
    // queries are coalesed into one big query if the current time has already
    // passed the scheduled time of multiple queries.
    if (scenario == TestScenario::Server &&
        settings.requested.server_coalesce_queries) {
      auto current_query_idx = queries_idx;
      for (; queries_idx + query_stride < queries_count;
           queries_idx += query_stride) {
        auto next_scheduled_time =
            start + queries[queries_idx + query_stride].scheduled_delta;
        // If current time hasn't reached the next query's scheduled time yet,
        // don't include next query.
        if (last_now < next_scheduled_time) {
          break;
        }
        queries_issued_per_iter++;
      }
      if (queries_idx > current_query_idx) {
        // Coalesced all the pass due queries.
        query.CoalesceQueries(queries.data(), current_query_idx + query_stride,
                              queries_idx, query_stride);
      }
    }

    // Issue the query to the SUT.
    {
      auto tracer3 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("IssueQuery"); });
      sut->IssueQuery(query.query_to_send);
    }

    // Increment the counter.
    expected_latencies += query.query_to_send.size();
    queries_issued_per_iter++;
    queries_issued += queries_issued_per_iter;

    if (scenario == TestScenario::Server &&
        settings.requested.server_coalesce_queries) {
      // Set the query back to its clean state.
      query.Decoalesce();
    }

    if (state->mode == TestMode::AccuracyOnly) {
      // TODO: Rate limit in accuracy mode so accuracy mode works even
      //       if the expected/target performance is way off.
      continue;
    }

    auto duration = (last_now - start);
    if (scenario == TestScenario::Server) {
      if (settings.max_async_queries != 0) {
        // Checks if there are too many outstanding queries.
        size_t queries_issued_total{0};
        if (multi_thread) {
          // To check actual number of async queries in multi-thread case,
          // we would have to combine the number of queries_issued from all
          // issue threads.
          {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->queries_issued += queries_issued_per_iter;
            queries_issued_total = state->queries_issued;
          }
        } else {
          queries_issued_total = queries_issued;
        }
        size_t queries_outstanding =
            queries_issued_total -
            response_logger.queries_completed.load(std::memory_order_relaxed);
        if (queries_outstanding > settings.max_async_queries) {
          LogDetail([thread_idx, queries_issued_total,
                     queries_outstanding](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
            std::stringstream ss;
            ss << "IssueQueryThread " << thread_idx
               << " Ending early: Too many outstanding queries."
               << " issued " << queries_issued_total << " outstanding "
               << queries_outstanding;
            MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
            detail.Error("IssueQueryThread ", std::to_string(thread_idx),
                         " Ending early: Too many outstanding queries.",
                         "issued", std::to_string(queries_issued_total),
                         "outstanding", std::to_string(queries_outstanding));
#endif
          });
          break;
        }
      }
    } else {
      // Checks if we end normally.
      if (queries_issued >= min_query_count_for_thread &&
          duration >= settings.target_duration) {
        LogDetail([thread_idx](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
          MLPERF_LOG(
              detail, "generic_message",
              "Ending naturally: Minimum query count and test duration met.");
#else
          detail(
              " Ending naturally: Minimum query count and test duration met.");
#endif
        });
        ran_out_of_generated_queries = false;
        break;
      }
    }

    // Checks if we have exceeded max_query_count for this thread.
    if (settings.max_query_count != 0 &&
        queries_issued >= max_query_count_for_thread) {
      LogDetail([thread_idx, queries_issued](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
        std::stringstream ss;
        ss << "IssueQueryThread " << thread_idx
           << " Ending early: Max query count reached."
           << " query_count " << queries_issued;
        MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
        detail.Error("IssueQueryThread ", std::to_string(thread_idx),
                     " Ending early: Max query count reached.", "query_count",
                     std::to_string(queries_issued));
#endif
      });
      ran_out_of_generated_queries = false;
      break;
    }

    // Checks if we have exceeded max_duration.
    if (settings.max_duration.count() != 0 &&
        duration > settings.max_duration) {
      LogDetail([thread_idx, duration](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
        std::stringstream ss;
        ss << "IssueQueryThread " << thread_idx
           << " Ending early: Max test duration reached."
           << " duration_ns " << duration.count();
        MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
        detail.Error("IssueQueryThread ", std::to_string(thread_idx),
                     " Ending early: Max test duration reached.", "duration_ns",
                     std::to_string(duration.count()));
#endif
      });
      ran_out_of_generated_queries = false;
      break;
    }
  }

  // Combine the issuing statistics from multiple issue threads.
  {
    std::lock_guard<std::mutex> lock(state->mtx);
    state->ran_out_of_generated_queries |= ran_out_of_generated_queries;
    // In Server scenario and when max_async_queries != 0, we would have set
    // state->queries_issued when we check max_async_queries in the loop.
    if (!(scenario == TestScenario::Server && settings.max_async_queries != 0 &&
          multi_thread)) {
      state->queries_issued += queries_issued;
    }
    state->expected_latencies += expected_latencies;
  }
}

}  // namespace loadgen

}  // namespace mlperf
