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
"""Generic tuning script."""

import argparse
import sys
from typing import Any


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
        help="Path to input model.",
    )
    parser.add_argument(
        "--output-graph",
        type=str,
        required=False,
        help="Output path for quantized model.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=False,
        help="Framework to use.",
    )
    return parser.parse_args()


def tune_model(
    input_graph: str,
    output_graph: str,
    config: str,
    framework: str,
) -> None:
    """Execute tuning."""
    from lpot.experimental import Quantization, common

    if framework == "onnxrt":
        import onnx

        input_graph = onnx.load(input_graph)

    quantizer = Quantization(config)
    quantizer.model = common.Model(input_graph)
    quantized_model = quantizer()
    if quantized_model is not None:
        quantized_model.save(output_graph)
    else:
        sys.exit(100)


def set_eager_execution(input_graph: str) -> None:
    """Set eager execution as required by model."""
    from lpot.ux.components.model.model_type_getter import get_model_type

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
    tune_model(
        input_graph=args.input_graph,
        output_graph=args.output_graph,
        config=args.config,
        framework=args.framework,
    )
