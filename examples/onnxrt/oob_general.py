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


from __future__ import absolute_import, division, print_function

import time
import logging
import argparse
import numpy as np
import onnx
import onnxruntime

from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def parse_dummy_input(model, benchmark_nums):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None)
    shapes = []
    lows = []
    highs = []
    dtypes = []
    len_inputs = len(session.get_inputs())
    for i in range(len_inputs):
        input = session.get_inputs()[i]
        input_shapes = input.shape
        # onnxruntime type like tensor(float), tensor(int32)...
        input_dtype = input.type[7:-1] if input.type != "tensor(float)" else "float32"
        shape = [benchmark_nums]
        for j in range(1, len(input_shapes)):
            input_shape = input_shapes[j]
            if type(input_shape) == str:
                shape.append(1)
            else:
                shape.append(input_shape)
        shapes.append(tuple(shape))
        lows.append(-2047. if "int" in input_dtype else -1.0)
        highs.append(2048. if "int" in input_dtype else 1.0)
        dtypes.append(input_dtype)
    return shapes, lows, highs, dtypes

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime general model quantization')
    parser = argparse.ArgumentParser(
    description='Evaluating ONNXRuntime general model quantization',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=8,
        help='Batch size for dev set and test set')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Pre-trained bert model onnx file.')
    parser.add_argument('--tune',action='store_true', default=False,
                        help='Get bert tuning quantization model with lpot.')
    parser.add_argument('--config',type=str, default=None,
                        help='Tuning config file path')
    parser.add_argument('--output_model',type=str, default=None,
                        help='output model path and name')
    parser.add_argument('--benchmark',action='store_true', default=False,
                        help='Get benchmark performance of quantized model.')
    parser.add_argument('--benchmark_nums', type=int, default=1000,
                        help="Benchmark numbers of samples")
    parser.add_argument('--accuracy_only', action='store_true', default=False,
                        help='Get dummy output loss of input_model')
    parser.add_argument('--input_shape', type=str,
                        help='Shapes of input to model, like 3x224x224 or 128x768, 128x256')
    args = parser.parse_args()
    model = onnx.load(args.model_path)

    shapes, lows, highs, dtypes = parse_dummy_input(model, args.benchmark_nums)

    if args.input_shape:
        input_shape = args.input_shape.replace(' ', '')
        input_shapes = input_shape.split(',')
        input_shapes = [input_shapes] if type(input_shapes)!=list else input_shapes
        input_shapes = [shape.split('x') for shape in input_shapes]
        shapes = [tuple([args.benchmark_nums] + [int(dim) for dim in shape]) for shape in input_shapes]

    from lpot.data.datasets.dummy_dataset import DummyDataset
    from lpot.data.dataloaders.onnxrt_dataloader import ONNXRTDataLoader
    dummy_dataset = DummyDataset(shapes, low=lows, high=highs, dtype=dtypes)
    dummy_dataloader = ONNXRTDataLoader(dummy_dataset, batch_size=args.eval_batch_size)

    def eval_func(model):
        return evaluate_onnxrt(model, dummy_dataloader, reference)

    if args.benchmark:
        from lpot import Benchmark
        evaluator = Benchmark(args.config)
        model = evaluator.model(model)
        results = evaluator(model=model, b_dataloader=dummy_dataloader)
        for mode, result in results.items():
            acc, batch_size, result_list = result
            latency = np.array(result_list).mean() / batch_size

            print('\n quantized model {} mode benchmark result:'.format(mode))
            print('Accuracy is {:.3f}'.format(acc))
            print('Batch size = {}'.format(batch_size))
            print('Latency: {:.3f} ms'.format(latency * 1000))
            print('Throughput: {:.3f} images/sec'.format(batch_size * 1./ latency))
    
    if args.tune:

        from lpot.quantization import Quantization
        quantize = Quantization(args.config)
        model = quantize.model(model)
        q_model = quantize(
            model, 
            q_dataloader=dummy_dataloader,
            eval_dataloader=dummy_dataloader)
        onnx.save(q_model, args.output_model)
