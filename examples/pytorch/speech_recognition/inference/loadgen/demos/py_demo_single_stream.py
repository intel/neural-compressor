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

"""Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function

import array
import threading
import time

from absl import app
import mlperf_loadgen
import numpy


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return


def process_query_async(query_samples):
    """Processes the list of queries."""
    time.sleep(.001)
    responses = []
    response_array = array.array(
        "f", [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 254, 255])
    response_info = response_array.buffer_info()
    response_data = response_info[0]
    response_size = response_info[1] * response_array.itemsize
    for s in query_samples:
        responses.append(
            mlperf_loadgen.QuerySampleResponse(
                s.id, response_data, response_size))
    mlperf_loadgen.QuerySamplesComplete(responses)


def issue_query(query_samples):
    threading.Thread(target=process_query_async,
                     args=[query_samples]).start()


def flush_queries():
    pass


def process_latencies(latencies_ns):
    print("Average latency: ")
    print(numpy.mean(latencies_ns))
    print("Median latency: ")
    print(numpy.percentile(latencies_ns, 50))
    print("90 percentile latency: ")
    print(numpy.percentile(latencies_ns, 90))


def main(argv):
    del argv
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.single_stream_expected_latency_ns = 1000000
    settings.min_query_count = 100
    settings.min_duration_ms = 10000

    sut = mlperf_loadgen.ConstructSUT(
        issue_query, flush_queries, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        1024, 128, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == "__main__":
    app.run(main)
