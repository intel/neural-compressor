# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Pruning optimization script."""

import argparse
import sys
from typing import Any


def parse_args() -> Any:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--input-graph",
        type=str,
        required=True,
        help="Path to input model.",
    )
    parser.add_argument(
        "--output-graph",
        type=str,
        required=True,
        help="Path to optimized model.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=False,
        help="Framework to use.",
    )
    return parser.parse_args()


def optimize_model(
    input_graph: str,
    output_graph: str,
    framework: str,
    config: str,
) -> None:
    """Execute pruning optimization."""
    import tensorflow as tf

    from neural_compressor.experimental import Pruning

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if framework == "onnxrt":
        import onnx

        input_graph = onnx.load(input_graph)

    prune = Pruning(config)
    prune.model = input_graph
    optimized_model = prune.fit()
    if optimized_model is not None:
        optimized_model.save(output_graph)
    else:
        sys.exit(100)

    stats, sparsity = optimized_model.report_sparsity()
    print(stats)
    print(sparsity)


if __name__ == "__main__":
    args = parse_args()

    optimize_model(
        input_graph=args.input_graph,
        output_graph=args.output_graph,
        config=args.config,
        framework=args.framework,
    )
