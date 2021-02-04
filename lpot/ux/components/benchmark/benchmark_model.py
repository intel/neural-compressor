# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic benchmark script."""

import argparse
import logging as log
from typing import Any, Dict, List

log.basicConfig(level=log.INFO)


def parse_args() -> Any:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--model", type=str, required=False, help="Path to model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="performance",
        choices=["accuracy", "performance"],
        help="Benchmark mode.",
    )
    return parser.parse_args()


def benchmark_model(
    model_path: str,
    config_path: str,
    benchmark_mode: str,
    datatype: str = "",
) -> List[Dict[str, Any]]:
    """Execute benchmark."""
    from lpot import Benchmark

    benchmark_results = []

    evaluator = Benchmark(config_path)
    results = evaluator(model=model_path)
    for mode, result in results.items():
        log.info(f"Mode: {mode}")
        acc, batch_size, result_list = result
        latency = (sum(result_list) / len(result_list)) / batch_size
        log.info(f"Batch size: {batch_size}")
        if mode == "accuracy":
            log.info(f"Accuracy: {acc:.3f}")
        elif mode == "performance":
            log.info(f"Latency: {latency * 1000:.3f} ms")
            log.info(f"Throughput: {1. / latency:.3f} images/sec")

        benchmark_results.append(
            {
                "precision": datatype,
                "mode": mode,
                "batch_size": batch_size,
                "accuracy": acc,
                "latency": latency * 1000,
                "throughput": 1.0 / latency,
            },
        )
    return benchmark_results


if __name__ == "__main__":
    args = parse_args()
    benchmark_model(
        model_path=args.model,
        config_path=args.yaml,
        benchmark_mode=args.mode,
    )
