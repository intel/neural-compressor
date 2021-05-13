# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

## \file
#  \brief MLPerf Inference LoadGen python module setup.
#  \details Creates a module that python can import.
#  All source files are compiled by python"s C++ toolchain  without depending
#  on a loadgen lib.
#
#  This setup.py can be used stand-alone, without the use of an external
#  build system. This will polute your source tree with output files
#  and binaries. Use one of the gn build targets instead if you want
#  to avoid poluting the source tree.

from setuptools import Extension
from setuptools import setup
from version_generator import generate_loadgen_version_definitions

generated_version_source_filename = "generated/version_generated.cc"
generate_loadgen_version_definitions(generated_version_source_filename,
                                     ".")

public_headers = [
    "loadgen.h",
    "query_sample.h",
    "query_sample_library.h",
    "system_under_test.h",
    "test_settings.h",
]

lib_headers = [
    "logging.h",
    "test_settings_internal.h",
    "trace_generator.h",
    "utils.h",
    "version.h",
]

lib_sources = [
    "issue_query_controller.cc",
    "loadgen.cc",
    "logging.cc",
    "test_settings_internal.cc",
    "utils.cc",
    "version.cc",
]

lib_bindings = [
    "bindings/python_api.cc",
]

mlperf_loadgen_headers = public_headers + lib_headers
mlperf_loadgen_sources_no_gen = lib_sources + lib_bindings
mlperf_loadgen_sources = (mlperf_loadgen_sources_no_gen +
                          [generated_version_source_filename])

mlperf_loadgen_module = Extension(
        "mlperf_loadgen",
        define_macros=[("MAJOR_VERSION", "0"), ("MINOR_VERSION", "5")],
        include_dirs=[".", "../third_party/pybind/include"],
        sources=mlperf_loadgen_sources,
        depends=mlperf_loadgen_headers)

setup(name="mlperf_loadgen",
      version="0.5a0",
      description="MLPerf Inference LoadGen python bindings",
      url="https://mlperf.org",
      ext_modules=[mlperf_loadgen_module])
