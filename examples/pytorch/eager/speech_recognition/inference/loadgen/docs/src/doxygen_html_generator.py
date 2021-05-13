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
#  \brief A script that sets the environment variables expected by doxygen.cfg.
#  \details This can be run manually without any arguments, but also allows a
#  build system to customize the output directory.

import os
import sys


def generate_doxygen_html(doxygen_out_dir, loadgen_root):
    os.environ["MLPERF_LOADGEN_SRC_PATH"] = loadgen_root
    os.environ["MLPERF_DOXYGEN_OUT_PATH"] = doxygen_out_dir
    os.popen("doxygen " + loadgen_root + "/docs/src/doxygen.cfg")


def main(argv):
    doxygen_out_dir = "./docs/gen" if len(argv) < 2 else argv[1]
    loadgen_root = "." if len(argv) < 3 else argv[2]
    generate_doxygen_html(doxygen_out_dir, loadgen_root)


main(sys.argv)
