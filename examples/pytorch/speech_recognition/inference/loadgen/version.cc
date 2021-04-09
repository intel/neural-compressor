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
/// \brief Non-generated version logic.

#include "version.h"

#include "logging.h"
#include "utils.h"

namespace mlperf {

/// Helper function to split a string based on a delimiting character.
std::vector<std::string> splitString(const std::string& input,
                                     const std::string& delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t next = 0;
  while (next != std::string::npos) {
    next = input.find(delimiter, start);
    result.emplace_back(input, start, next - start);
    start = next + 1;
  }
  return result;
}

/// Converts the hash-filename pairs to a dict.
std::map<std::string, std::string> LoadgenSha1OfFilesToDict(
    const std::string& in) {
  std::map<std::string, std::string> result;
  auto files = splitString(in, "\n");
  for (const auto& file : files) {
    auto hash_and_name = splitString(file, " ");
    assert(hash_and_name.size() > 1);
    result[hash_and_name[1]] = hash_and_name[0];
  }
  return result;
}

void LogLoadgenVersion() {
  LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "loadgen_version",
               LoadgenVersion() + " @ " + LoadgenGitRevision());
    MLPERF_LOG(detail, "loadgen_build_date_local", LoadgenBuildDateLocal());
    MLPERF_LOG(detail, "loadgen_build_date_utc", LoadgenBuildDateUtc());
    MLPERF_LOG(detail, "loadgen_git_commit_date", LoadgenGitCommitDate());
    MLPERF_LOG(detail, "loadgen_git_log_message",
               EscapeStringJson(LoadgenGitLog()));
    MLPERF_LOG(detail, "loadgen_git_status_message",
               EscapeStringJson(LoadgenGitStatus()));
    if (!LoadgenGitStatus().empty() && LoadgenGitStatus() != "NA") {
      MLPERF_LOG_ERROR(detail, "error_uncommitted_loadgen_changes",
                       "Loadgen built with uncommitted changes!");
      ;
    }
    MLPERF_LOG(detail, "loadgen_file_sha1",
               LoadgenSha1OfFilesToDict(LoadgenSha1OfFiles()));
#else
    detail("LoadgenVersionInfo:");
    detail("version : " + LoadgenVersion() + " @ " + LoadgenGitRevision());
    detail("build_date_local : " + LoadgenBuildDateLocal());
    detail("build_date_utc   : " + LoadgenBuildDateUtc());
    detail("git_commit_date  : " + LoadgenGitCommitDate());
    detail("git_log :\n\n" + LoadgenGitLog() + "\n");
    detail("git_status :\n\n" + LoadgenGitStatus() + "\n");
    if (!LoadgenGitStatus().empty() && LoadgenGitStatus() != "NA") {
      detail.Error("Loadgen built with uncommitted changes!");
    }
    detail("SHA1 of files :\n\n" + LoadgenSha1OfFiles() + "\n");
#endif
  });
}

}  // namespace mlperf
