"""
MRPC with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained MobileBERT parameters for
for Microsoft Research Paraphrase Corpus (MRPC) task.

@article{sun2020mobilebert,
  title={Mobilebert: a compact task-agnostic bert for resource-limited devices},
  author={Sun, Zhiqing and Yu, Hongkun and Song, Xiaodan and Liu, Renjie and Yang, Yiming and Zhou, Denny},
  journal={arXiv preprint arXiv:2004.02984},
  year={2020}
}
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import logging
import argparse
import onnx

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime full precision accuracy and performance:')
    parser = argparse.ArgumentParser(
    description='MobileBERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained resnet50 model on onnx file"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="config yaml path"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    args = parser.parse_args()

    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        model = onnx.load(args.model_path)
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator(args.mode)

    if args.tune:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_embed_layer_norm = False

        model_optimizer = optimizer.optimize_model(
            args.model_path,
            'bert',
            num_heads=4,
            hidden_size=512,
            optimization_options=opt_options)
        model = model_optimizer.model

        from neural_compressor.experimental import Quantization, common
        from neural_compressor import options
        options.onnxrt.qdq_setting.OpTypesToExcludeOutputQuantizatioin = ['MatMul']
        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        q_model = quantize()
        q_model.save(args.output_model)
