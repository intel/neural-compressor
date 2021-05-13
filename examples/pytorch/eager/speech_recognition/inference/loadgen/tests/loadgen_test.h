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
/// \brief A minimal test framework.

#ifndef MLPERF_LOADGEN_TESTS_LOADGEN_TEST_H_
#define MLPERF_LOADGEN_TESTS_LOADGEN_TEST_H_

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#define REGISTER_TEST(name, ...) \
  static Test::StaticRegistrant test##name(#name, __VA_ARGS__);

#define REGISTER_TEST_SCENARIO(name, scenario, test, ...) \
  static Test::StaticRegistrant t##name##scenario(        \
      #name "_" #scenario, test, __VA_ARGS__, mlperf::TestScenario::scenario)

#define REGISTER_TEST_ALL_SCENARIOS(name, test, ...)                \
  REGISTER_TEST_SCENARIO(name, SingleStream, test, __VA_ARGS__);    \
  REGISTER_TEST_SCENARIO(name, MultiStream, test, __VA_ARGS__);     \
  REGISTER_TEST_SCENARIO(name, MultiStreamFree, test, __VA_ARGS__); \
  REGISTER_TEST_SCENARIO(name, Server, test, __VA_ARGS__);          \
  REGISTER_TEST_SCENARIO(name, Offline, test, __VA_ARGS__);

#define FAIL_IF(exp)                                              \
  [&]() {                                                         \
    const bool v = exp;                                           \
    if (v) {                                                      \
      std::cerr << "\n   ERROR: (" << __FILE__ << "@" << __LINE__ \
                << ") : " #exp;                                   \
      Test::AddFailure();                                         \
    }                                                             \
    return v;                                                     \
  }()

#define FAIL_MSG(...)                                                      \
  [&]() {                                                                  \
    std::cerr << "\n    Info: (" << __FILE__ << "@" << __LINE__ << ") : "; \
    Test::Log(__VA_ARGS__);                                                \
    return true;                                                           \
  }()

#define FAIL_EXP(exp)                                                      \
  [&]() {                                                                  \
    std::cerr << "\n    Info: (" << __FILE__ << "@" << __LINE__ << ") : "; \
    std::cerr << #exp << " is " << (exp);                                  \
    return true;                                                           \
  }()

#define BAD_TEST_MSG(...)                                        \
  [&]() {                                                        \
    FAIL_MSG("The test isn't testing what it claims to test. "); \
    Test::Log(__VA_ARGS__);                                      \
    return true;                                                 \
  }()

#define ABORT_TEST                                     \
  [&]() {                                              \
    FAIL_MSG("ABORTING");                              \
    throw std::logic_error("ABORT_TEST encountered."); \
    return false;                                      \
  }();

/// \brief Testing utilities.
namespace testing {

/// \brief Wraps a test class as a functor for easy registration.
/// Forwards registration args to a SetUpTest method.
/// \details Calls SetUpTest, RunTest, and EndTest.
template <typename TestT>
struct TestProxy {
  template <typename... Args>
  void operator()(Args&&... args) {
    TestT test;
    test.SetUpTest(std::forward<Args>(args)...);
    test.RunTest();
    test.EndTest();
  }
};

/// \brief A collection of methods for registering and running tests.
class Test {
  /// \brief Maps registered test names to a callback.
  using TestMap = std::multimap<const char*, std::function<void()>>;

  /// \brief The registered tests.
  /// \details Wraps a static local to avoid undefined initialization order
  /// and guarantee it is initialized before the first test registers itself.
  static TestMap& tests() {
    static TestMap tests_;
    return tests_;
  }

  /// \brief The number of errors the current test has encountered.
  static size_t& test_fails() {
    static size_t test_fails_ = 0;
    return test_fails_;
  }

 public:
  /// \brief Registers a test before main() starts during static initialization.
  struct StaticRegistrant {
    template <typename... Args>
    StaticRegistrant(Args&&... args) {
      Test::Register(std::forward<Args>(args)...);
    }
  };

  /// \brief Registers a test at runtime.
  template <typename TestF, typename... Args>
  static void Register(const char* name, TestF test, Args&&... args) {
    std::function<void()> test_closure =
        std::bind(test, std::forward<Args>(args)...);
    tests().insert({std::move(name), std::move(test_closure)});
  }

  /// \brief Runs all currently registered tests that match the given filter.
  static int Run(std::function<bool(const char*)> filter) {
    // Determine which tests are enabled.
    std::vector<TestMap::value_type*> enabled_tests;
    for (auto& test : tests()) {
      if (filter(test.first)) {
        enabled_tests.push_back(&test);
      }
    }
    const size_t enabled = enabled_tests.size();
    std::cout << enabled << " of " << tests().size() << " tests enabled.\n";

    // Run the tests.
    std::vector<const char*> failures;
    for (size_t i = 0; i < enabled; i++) {
      const char* name = enabled_tests[i]->first;
      std::cout << "[" << (i + 1) << "/" << enabled << "] : " << name << " : ";
      std::cout.flush();
      test_fails() = 0;
      try {
        enabled_tests[i]->second();  // Run the test.
      } catch (std::exception& e) {
        constexpr bool TestThrewException = true;
        FAIL_IF(TestThrewException) && FAIL_EXP(e.what());
      }
      if (test_fails() > 0) {
        failures.push_back(name);
        std::cerr << "\n FAILED: " << name << "\n";
      } else {
        std::cout << "SUCCESS\n";
      }
    }

    // Summarize.
    if (enabled_tests.empty()) {
      std::cerr << "Check your test filter.\n";
    } else if (failures.empty()) {
      std::cout << "All " << enabled << " tests passed! \\o/\n";
    } else {
      std::cout << failures.size() << " of " << enabled << " tests failed:\n";
      for (auto failed_test_name : failures) {
        std::cout << "  " << failed_test_name << "\n";
      }
    }
    return failures.size();
  }

  /// \brief Used by test macros to flag test failure.
  static void AddFailure() { test_fails()++; }

  /// \brief Base case for the variadic version of Log.
  static void Log() {}

  /// \brief Used by test macros to log an arbitrary list of args.
  template <typename T, typename... Args>
  static void Log(T&& v, Args&&... args) {
    std::cerr << v;
    Log(std::forward<Args>(args)...);
  }
};

}  // namespace testing

// The testing namespace exists for documentation purposes.
// Export the testing namespace for all files that define tests.
using namespace testing;

#endif  // MLPERF_LOADGEN_TESTS_LOADGEN_TEST_H_
