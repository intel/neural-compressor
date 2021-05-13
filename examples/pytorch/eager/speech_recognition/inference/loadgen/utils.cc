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

#include "utils.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "logging.h"

namespace mlperf {

std::string DoubleToString(double value, int precision) {
  std::stringstream ss;
  ss.precision(precision);
  ss << std::fixed << value;
  return ss.str();
}

bool FileExists(const std::string filename) {
  std::ifstream file_object(filename);
  return file_object.good();
}

namespace {

std::string DateTimeString(const char* format,
                           std::chrono::system_clock::time_point tp,
                           bool append_ms) {
  std::time_t tp_time_t = std::chrono::system_clock::to_time_t(tp);
  std::tm date_time = *std::localtime(&tp_time_t);
  constexpr size_t kDateTimeMaxSize = 256;
  char date_time_cstring[kDateTimeMaxSize];
  std::strftime(date_time_cstring, kDateTimeMaxSize, format, &date_time);
  std::string date_time_string(date_time_cstring);
  if (!append_ms) {
    return date_time_string;
  }

  auto tp_time_t_part = std::chrono::system_clock::from_time_t(tp_time_t);
  auto tp_remainder = tp - tp_time_t_part;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp_remainder)
                .count();
  if (ms < 0 || ms >= 1000) {
    LogDetail([ms](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "WARNING: Unexpected milliseconds getting date and time."
         << " ms: " << ms;
      MLPERF_LOG_WARNING(detail, "warning_generic_message", ss.str());
#else
      detail("WARNING: Unexpected milliseconds getting date and time.", "ms",
             ms);
#endif
    });
  }
  std::string ms_string = std::to_string(ms);
  // Prefix with zeros so length is always 3.
  ms_string.insert(0, std::min<size_t>(2, 3 - ms_string.length()), '0');
  return date_time_string + "." + ms_string;
}

}  // namespace

std::string CurrentDateTimeISO8601() {
  return DateTimeString("%FT%TZ", std::chrono::system_clock::now(), false);
}

std::string DateTimeStringForPower(std::chrono::system_clock::time_point tp) {
  return DateTimeString("%m-%d-%Y %T", tp, true);
}

std::string EscapeStringJson(const std::string& in) {
  std::stringstream ss;
  for (auto c = in.cbegin(); c != in.cend(); c++) {
    int c_val = static_cast<int>(*c);
    switch (*c) {
      case '"':
        ss << "\\\"";
        break;
      case '\\':
        ss << "\\\\";
        break;
      case '\b':
        ss << "\\b";
        break;
      case '\f':
        ss << "\\f";
        break;
      case '\n':
        ss << "\\n";
        break;
      case '\r':
        ss << "\\r";
        break;
      case '\t':
        ss << "\\t";
        break;
      default:
        if (c_val >= 0x00 && c_val < 0x20) {
          ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << c_val;
        } else {
          ss << *c;
        }
    }
  }
  return ss.str();
}

}  // namespace mlperf
