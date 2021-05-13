#!/usr/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
echo "Building loadgen in Debug mode..."
if [ ! -e loadgen_build ]; then mkdir loadgen_build; fi;
cd loadgen_build && cmake -DCMAKE_BUILD_TYPE=Debug ../.. && make -j && cd ..
echo "Building test program in Debug mode..."
if [ ! -e build ]; then mkdir build; fi;
g++ --std=c++11 -O0 -g -I.. -o build/repro.exe repro.cpp -Lloadgen_build -lmlperf_loadgen -lpthread && \
gdb --args build/repro.exe $1 $2 $3 $4 $5 $6
