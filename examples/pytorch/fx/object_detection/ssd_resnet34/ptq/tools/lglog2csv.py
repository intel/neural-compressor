"""
collect mlperf loadgen output to csv
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import re
import time


# pylint: disable=missing-docstring


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input json")
    parser.add_argument("--runtime", required=True, help="runtime")
    parser.add_argument("--machine", required=True, help="machine")
    parser.add_argument("--model", required=True, help="model")
    parser.add_argument("--name", required=True, help="name")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # print("name,date,machine,runtime,model,mode,qps,mean,latency_90,latency_99")

    now = int(time.time())

    with open(args.input, "r") as fp:
        mode, mean, latency_90, latency_99, qps = None, 0, 0, 0, 0
        for line in fp:
            m = re.match("^Scenario\s*:\s*(\w+).*", line)
            if m:
                mode = m.group(1)
            m = re.match("^90.00 percentile latency.*:\s*(\d+).*", line)
            if m:
                latency_90 = m.group(1)
            m = re.match("^99.00 percentile latency.*:\s*(\d+).*", line)
            if m:
                latency_99 = m.group(1)
            m = re.match("^Mean latency.*:\s*(\d+).*", line)
            if m:
                mean = m.group(1)
            m = re.match("^Completed samples per second.*:\s*(\d+).*", line)
            if m:
                qps = m.group(1)
            m = re.match("^QPS w/ loadgen overhead.*:\s*(\d+).*", line)
            if m:
                qps = m.group(1)
            m = re.match("^Samples per second.*:\s*(\d+).*", line)
            if m:
                qps = m.group(1)
            m = re.match("Test Parameters Used.*", line)
            if m:
                print("{},{},{},{},{},{},{},{},{},{}".format(
                    args.name, now, args.machine, args.runtime, args.model,
                    mode, qps, mean, latency_90, latency_99))
                mode, mean, latency_90, latency_99, qps = None, 0, 0, 0, 0


if __name__ == "__main__":
    main()
