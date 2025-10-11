# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)


class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--model", "--model_name", "--model_name_or_path",
                          help="model name or path")

        self.add_argument('--scheme', default="MXFP4", type=str,
                          help="quantizaion scheme.")

        self.add_argument("--device", "--devices", default="auto", type=str,
                          help="the device to be used for tuning. The default is set to auto,"
                               "allowing for automatic detection."
                               "Currently, device settings support CPU, GPU, and HPU.")

        self.add_argument("--export_format", default="llm_compressor", type=str,
                          help="the format to save the model"
                          )

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                          help="the directory to save quantized model")

        self.add_argument("--fp_layers", default="", type=str,
                          help="layers to maintain original data type")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--iters", "--iter", default=0, type=int,
                        help=" iters")

    args = parser.parse_args()
    return args


def tune(args):
    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(f"start to quantize {model_name}")

    layer_config = {}
    model = Llama4ForConditionalGeneration.from_pretrained(args.model, device_map=None, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fp_layers = args.fp_layers.replace(" ", "").split(",")
    if len(fp_layers) > 0:
        for n, m in model.named_modules():
            if not isinstance(m, (torch.nn.Linear)):
                continue
            for name in fp_layers:
                if name in n:
                    layer_config[n] = {"bits": 16, "act_bits": 16}
                    break

    qconfig = AutoRoundConfig(
        tokenizer=tokenizer,
        scheme=args.scheme,
        layer_config=layer_config,
        export_format="llm_compressor",
        is_mllm=True,
        output_dir=args.output_dir,
    )
    model = prepare(model, qconfig)
    model = convert(model, qconfig)

if __name__ == '__main__':
    args = setup_parser()
    tune(args)
