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
/// \brief A main entry point a test binary can use if it just wants to execute
/// Test::Run on all statically registered tests.

#include <regex>

#include "loadgen_test.h"

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " <include_regex> <exclude_regex>\n";
    return -1;
  }
  std::regex include_regex(argc >= 2 ? argv[1] : ".*");
  std::regex exclude_regex(argc >= 3 ? std::regex(argv[2]) : std::regex());
  auto test_filter = [&](const char* test_name) {
    return (std::regex_search(test_name, include_regex) &&
            !std::regex_search(test_name, exclude_regex));
  };
  return Test::Run(test_filter);
}
