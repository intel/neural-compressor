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
echo "Building loadgen..."
if [ ! -e loadgen_build ]; then mkdir loadgen_build; fi;
cd loadgen_build && cmake ../.. && make -j && cd ..
echo "Building test program..."
if [ ! -e build ]; then mkdir build; fi;
g++ --std=c++11 -O3 -I.. -o build/repro.exe repro.cpp -Lloadgen_build -lmlperf_loadgen -lpthread && \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 build/repro.exe $1 $2 $3 $4 $5 $6
