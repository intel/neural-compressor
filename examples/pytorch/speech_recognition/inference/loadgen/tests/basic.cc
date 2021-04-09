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
/// \brief Basic functionality unit tests.

#include <algorithm>
#include <deque>
#include <future>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

#include "../loadgen.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"
#include "loadgen_test.h"

/// \brief Correctness unit tests.
namespace unit_tests {

/// \defgroup LoadgenTestsBasic Test Coverage: Basic

/// \brief Implements the client interfaces of the loadgen and
/// has some basic sanity checks that are enabled for all tests.
/// \details It also forwards calls to overrideable *Ext methods and implements
/// the TestProxy concept.
struct SystemUnderTestBasic : public mlperf::QuerySampleLibrary,
                              public mlperf::SystemUnderTest {
  const std::string& Name() const override { return name_; }

  size_t TotalSampleCount() override { return total_sample_count_; }
  size_t PerformanceSampleCount() override { return performance_sample_count_; }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto s : samples) {
      samples_load_count_.at(s)++;
      loaded_samples_.push_back(s);
    }
    LoadSamplesToRamExt(samples);
  }
  virtual void LoadSamplesToRamExt(
      const std::vector<mlperf::QuerySampleIndex>& samples) {}

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto s : samples) {
      FAIL_IF(loaded_samples_.front() != s) &&
          FAIL_EXP(loaded_samples_.front()) && FAIL_EXP(s);
      loaded_samples_.pop_front();
      size_t prev_load_count = samples_load_count_.at(s)--;
      FAIL_IF(prev_load_count == 0) && FAIL_EXP(prev_load_count);
    }
    UnloadSamplesFromRamExt(samples);
  }
  virtual void UnloadSamplesFromRamExt(
      const std::vector<mlperf::QuerySampleIndex>& samples) {}

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    query_sizes_.push_back(samples.size());
    samples_between_flushes_.back() += samples.size();
    responses.reserve(samples.size());
    for (auto s : samples) {
      FAIL_IF(samples_load_count_.at(s.index) == 0) &&
          FAIL_MSG("Issued unloaded sample:") && FAIL_EXP(s.index);
      samples_issue_count_.at(s.index)++;
      issued_samples_.push_back(s.index);
      responses.push_back({s.id, 0, 0});
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
    IssueQueryExt(samples);
  }
  virtual void IssueQueryExt(const std::vector<mlperf::QuerySample>& samples) {}

  void FlushQueries() override {
    samples_between_flushes_.push_back(0);
    FlushQueriesExt();
  }
  virtual void FlushQueriesExt() {}

  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

  virtual void RunTest() {
    samples_load_count_.resize(total_sample_count_, 0);
    samples_issue_count_.resize(total_sample_count_, 0);
    samples_between_flushes_.resize(1, 0);
    mlperf::StartTest(this, this, test_settings_, log_settings_);
  }

  virtual void EndTest() {}

 protected:
  mlperf::TestSettings test_settings_;
  mlperf::LogSettings log_settings_;

  std::string name_{"BasicSUT"};
  size_t total_sample_count_;
  size_t performance_sample_count_;
  std::vector<mlperf::QuerySampleIndex> issued_samples_;
  std::deque<mlperf::QuerySampleIndex> loaded_samples_;
  std::vector<size_t> samples_load_count_;
  std::vector<size_t> samples_issue_count_;

  std::vector<size_t> query_sizes_;
  std::vector<size_t> samples_between_flushes_;
};

/// \brief Provides common test set up logic.
struct SystemUnderTestAccuracy : public SystemUnderTestBasic {
  virtual void SetUpTest(size_t samples_per_query,
                         size_t samples_per_query_remainder,
                         size_t accuracy_remainder,
                         mlperf::TestScenario scenario) {
    performance_sample_count_ =
        samples_per_query * 16 + samples_per_query_remainder;
    total_sample_count_ = performance_sample_count_ * 32 + accuracy_remainder;

    log_settings_.log_output.prefix_with_datetime = false;

    test_settings_.scenario = scenario;
    test_settings_.mode = mlperf::TestMode::AccuracyOnly;
    test_settings_.multi_stream_samples_per_query = samples_per_query;

    double qps = 1e3;
    test_settings_.server_target_qps = qps;
    test_settings_.multi_stream_target_qps = qps;
  }
};

/// \brief Verifies all samples from the QSL are included at least once
/// in accuracy mode.
/// \ingroup LoadgenTestsBasic
struct TestAccuracyIncludesAllSamples : public SystemUnderTestAccuracy {
  void EndTest() override {
    std::sort(issued_samples_.begin(), issued_samples_.end());

    FAIL_IF(issued_samples_.size() < total_sample_count_) &&
        FAIL_EXP(issued_samples_.size()) && FAIL_EXP(total_sample_count_);
    FAIL_IF(issued_samples_.front() != 0) && FAIL_EXP(issued_samples_.front());
    FAIL_IF(issued_samples_.back() != total_sample_count_ - 1) &&
        FAIL_EXP(issued_samples_.back()) && FAIL_EXP(total_sample_count_);

    mlperf::QuerySampleIndex prev = -1;
    size_t discontinuities = 0;
    size_t dupes = 0;
    for (auto s : issued_samples_) {
      if (s == prev) {
        dupes++;
      } else if (s - prev > 1) {
        discontinuities++;
      }
      prev = s;
    }

    FAIL_IF(discontinuities != 0) && FAIL_EXP(discontinuities);
    if (test_settings_.scenario == mlperf::TestScenario::MultiStream ||
        test_settings_.scenario == mlperf::TestScenario::MultiStreamFree) {
      const size_t expected_sets =
          total_sample_count_ / performance_sample_count_;
      FAIL_IF(dupes >=
              test_settings_.multi_stream_samples_per_query * expected_sets) &&
          FAIL_EXP(dupes);
    } else {
      FAIL_IF(dupes != 0) && FAIL_EXP(dupes);
    }
  }
};

REGISTER_TEST_ALL_SCENARIOS(AccuracyIncludesAllSamples,
                            TestProxy<TestAccuracyIncludesAllSamples>(), 4, 0,
                            0);

/// \brief Verifies samples from the QSL aren't included too many times.
/// \details This is a regression test for:
/// https://github.com/mlperf/inference/pull/386
/// The root cause was using different values for samples_per_query while
/// generating queries for the GNMT dataset.
/// \ingroup LoadgenTestsBasic
struct TestAccuracyDupesAreLimitted : public SystemUnderTestAccuracy {
  void SetUpTest(bool, mlperf::TestScenario scenario) {
    SystemUnderTestAccuracy::SetUpTest(4, 0, 0, scenario);
    total_sample_count_ = 3003;
    performance_sample_count_ = 1001;
  }

  void EndTest() override {
    std::sort(issued_samples_.begin(), issued_samples_.end());

    FAIL_IF(issued_samples_.size() < total_sample_count_) &&
        FAIL_EXP(issued_samples_.size()) && FAIL_EXP(total_sample_count_);
    FAIL_IF(issued_samples_.front() != 0) && FAIL_EXP(issued_samples_.front());
    FAIL_IF(issued_samples_.back() != total_sample_count_ - 1) &&
        FAIL_EXP(issued_samples_.back()) && FAIL_EXP(total_sample_count_);

    std::vector<size_t> issue_counts(total_sample_count_, 0);
    for (auto s : issued_samples_) {
      issue_counts.at(s)++;
    }

    const bool multistream =
        test_settings_.scenario == mlperf::TestScenario::MultiStream ||
        test_settings_.scenario == mlperf::TestScenario::MultiStreamFree;
    const size_t max_count = multistream ? 2 : 1;

    for (size_t i = 0; i < issue_counts.size(); i++) {
      FAIL_IF(issue_counts[i] > max_count) && FAIL_EXP(i) &&
          FAIL_EXP(max_count) && FAIL_EXP(issue_counts[i]);
    }
  }
};

REGISTER_TEST_ALL_SCENARIOS(TestAccuracyDupesAreLimitted,
                            TestProxy<TestAccuracyDupesAreLimitted>(), true);

/// \brief Verifies offline + accuracy doesn't hang if the last set
/// in the accuracy series is smaller than others.
/// \ingroup LoadgenTestsBasic
struct TestOfflineRemainderAccuracySet : public SystemUnderTestAccuracy {
  void SetUpTest() {
    SystemUnderTestAccuracy::SetUpTest(4, 0, 7, mlperf::TestScenario::Offline);
  }

  void EndTest() override {
    auto& flush_samples = samples_between_flushes_;

    FAIL_IF(flush_samples.size() < 3) && FAIL_EXP(flush_samples.size()) &&
        BAD_TEST_MSG("Test should generate multiple query sets.") && ABORT_TEST;

    // The last counter will be 0, since a test ends with a call to
    // FlushQuery.
    FAIL_IF(flush_samples.back() != 0) && FAIL_EXP(flush_samples.back()) &&
        FAIL_MSG(
            "Detected stray calls to IssueQuery after the last call to "
            "FlushQuery.");
    flush_samples.pop_back();

    // Verify the test ran with a smaller last accuracy set.
    size_t first_size = flush_samples.front();
    size_t last_size = flush_samples.back();
    FAIL_IF(first_size <= last_size) && FAIL_EXP(first_size) &&
        FAIL_EXP(last_size) && BAD_TEST_MSG();

    flush_samples.pop_back();  // Don't check the last set for equality.
    for (size_t query_size : flush_samples) {
      FAIL_IF(query_size != first_size) && FAIL_EXP(query_size) &&
          FAIL_EXP(first_size);
    }
  }
};

REGISTER_TEST(Offline_RemainderAccuracySets,
              TestProxy<TestOfflineRemainderAccuracySet>());

/// \brief Verifies all queries only contain samples that are contiguous,
/// even if the set size is not a multiple of samples_per_query.
/// \ingroup LoadgenTestsBasic
struct TestMultiStreamContiguousRemainderQuery
    : public SystemUnderTestAccuracy {
  void SetUpTest(mlperf::TestScenario scenario) {
    SystemUnderTestAccuracy::SetUpTest(4, 1, 0, scenario);
    first_qsl_offsets_.resize(total_sample_count_, kBadQslOffset);

    auto spq = test_settings_.multi_stream_samples_per_query;
    FAIL_IF(performance_sample_count_ % spq == 0) &&
        FAIL_EXP(performance_sample_count_) && FAIL_EXP(spq) &&
        BAD_TEST_MSG("There is no remainder.");
  }

  void LoadSamplesToRamExt(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    FAIL_IF(loaded_samples_.size() != samples.size()) &&
        FAIL_MSG("Contiguous sample order is likely ambiguous.");
    for (size_t i = 0; i < samples.size(); i++) {
      auto& offset = first_qsl_offsets_.at(samples.at(i));
      // Samples may be loaded into multiple slots for paddign purposes,
      // so make sure to only index the first time a sample appears in a
      // loaded set.
      if (offset == kBadQslOffset) {
        offset = i;
      }
    }
  }

  void UnloadSamplesFromRamExt(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    FAIL_IF(!loaded_samples_.empty()) &&
        FAIL_MSG("Contiguous sample order is likely ambiguous.");
    for (size_t i = 0; i < samples.size(); i++) {
      first_qsl_offsets_.at(samples.at(i)) = kBadQslOffset;
    }
  }

  void IssueQueryExt(const std::vector<mlperf::QuerySample>& samples) override {
    size_t expected_offset = first_qsl_offsets_[samples[0].index];
    for (auto s : samples) {
      FAIL_IF(loaded_samples_[expected_offset] != s.index) &&
          FAIL_MSG("Samples are not contiguous.");
      expected_offset++;
    }
  }

  void FlushQueriesExt() override {}

  void EndTest() override {}

 private:
  static const size_t kBadQslOffset;
  std::vector<size_t> first_qsl_offsets_;
};

constexpr size_t TestMultiStreamContiguousRemainderQuery::kBadQslOffset =
    std::numeric_limits<size_t>::max();

REGISTER_TEST(MultiStream_RemainderQueryContiguous,
              TestProxy<TestMultiStreamContiguousRemainderQuery>(),
              mlperf::TestScenario::MultiStream);
REGISTER_TEST(MultiStreamFree_RemainderQueryContiguous,
              TestProxy<TestMultiStreamContiguousRemainderQuery>(),
              mlperf::TestScenario::MultiStreamFree);

}  // namespace unit_tests
