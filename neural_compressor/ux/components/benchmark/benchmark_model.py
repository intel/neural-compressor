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
from typing import Any

from neural_compressor.ux.components.benchmark import Benchmarks


def parse_args() -> Any:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--input-graph",
        type=str,
        required=False,
        help="Path to model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=Benchmarks.PERF,
        choices=[Benchmarks.PERF, Benchmarks.ACC],
        help="Benchmark mode.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=False,
        help="Framework to use.",
    )
    return parser.parse_args()


def benchmark_model(
    input_graph: str,
    config: str,
    benchmark_mode: str,
    framework: str,
) -> None:
    """Execute benchmark."""
    from neural_compressor.experimental import Benchmark, common

    if framework == "onnxrt":
        import onnx

        input_graph = onnx.load(input_graph)

    evaluator = Benchmark(config)
    evaluator.model = common.Model(input_graph)
    evaluator(benchmark_mode)


def set_eager_execution(input_graph: str) -> None:
    """Set eager execution as required by model."""
    from neural_compressor.ux.components.model.model_type_getter import get_model_type

    model_type = get_model_type(input_graph)

    try:
        import tensorflow as tf

        if "keras" == model_type:
            tf.compat.v1.enable_eager_execution()
        else:
            tf.compat.v1.disable_eager_execution()
    except Exception as err:
        print(err)


if __name__ == "__main__":
    args = parse_args()
    set_eager_execution(args.input_graph)
    benchmark_model(
        input_graph=args.input_graph,
        config=args.config,
        benchmark_mode=args.mode,
        framework=args.framework,
    )
