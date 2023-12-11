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

import numpy as np
import onnx

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

class Dataloader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        shape = [[batch_size, 4, 64, 64], [batch_size], [batch_size, 77, 768]]
        dtype = ['float32', 'float32', 'float32']
        self.dataset = []
        for idx in range(0, len(shape)):
            tensor = np.random.uniform(size=shape[idx])
            tensor = tensor.astype(dtype[idx])
            self.dataset.append(tensor)

    def __iter__(self):
         yield self.dataset, 0

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained model on onnx file"
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
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    parser.add_argument(
        '--quant_format',
        type=str,
        default='default', 
        choices=['default', 'QDQ', 'QOperator'],
        help="quantization format"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    args = parser.parse_args()

    dataloader = Dataloader(args.batch_size)

    if args.benchmark and args.mode == 'performance':
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        conf = BenchmarkConfig(warmup=10, iteration=1000, cores_per_instance=4, num_of_instance=1)
        fit(args.model_path, conf, b_dataloader=dataloader)
    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(quant_format=args.quant_format, recipes={'graph_optimization_level':'ENABLE_EXTENDED'})
        q_model = quantization.fit(args.model_path, config, calib_dataloader=dataloader)

        q_model.save(args.output_model)
