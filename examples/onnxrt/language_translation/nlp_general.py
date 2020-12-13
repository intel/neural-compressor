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

def parse_dummy_input(model, benchmark_nums, max_seq_length):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None)
    shapes = []
    lows = []
    highs = []
    for i in range(len(session.get_inputs())):
        input_name = session.get_inputs()[i].name
        input_shapes = session.get_inputs()[i].shape
        shape = [benchmark_nums]
        for input_shape in input_shapes:
            if 'seq' in input_shape :
                shape.append(max_seq_length)
            if input_name == "input_ids":
                low = 0.0
                high = 1000.0
            else:
                low = 0.0
                high = 2.0
        shapes.append(tuple(shape))
        lows.append(low)
        highs.append(high)
    return shapes, lows, highs

def evaluate_onnx(model, dataloader):

    session = onnxruntime.InferenceSession(model.SerializeToString(), None)

    len_inputs = len(session.get_inputs())
    for batch in tqdm(dataloader, desc="Evaluating"):
        ort_inputs = {
                        session.get_inputs()[i].name:  batch[i] for i in range(len_inputs)
                        }
        session.run(None, ort_inputs)
    return 1

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime full precision accuracy and performance:')
    parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size. Number of examples per gpu in a minibatch.')
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=8,
        help='Batch size for dev set and test set')
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='report interval')
    parser.add_argument(
        '--seed', type=int, default=2, help='Random seed')
    parser.add_argument(
        '--gpu', type=int, default=None, help='Which gpu for finetuning.')
    parser.add_argument(
        '--task_name',
        type=str,
        choices=['mnli', 'mrpc'],
        help='The name of the task to fine-tune. Choices include MRPC, QQP, '
            'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
    parser.add_argument(
        '--bert_dataset',
        type=str,
        default='MRPC',
        choices=['MRPC'],
        help='The dataset BERT pre-trained with.')
    parser.add_argument(
        '--model_type',
        type=str,
        default='bert')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Pre-trained bert model onnx file.')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./',
        help='The input directory where the model params are stored.')
    parser.add_argument(
        '--only_inference',
        action='store_true',
        help='If set, we skip training and only perform inference on dev and test data.')
    parser.add_argument('--max_seq_length', type=int,
                        default=128,
                        help='max seq length')
    parser.add_argument('--model_name_or_path', type=str,
                        default='bert-base-uncased',
                        help='model name or path')
    parser.add_argument('--data_dir', type=str,
                        help='datseset path')
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
    args = parser.parse_args()
    model = onnx.load(args.model_path)
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False

    model_optimizer = optimizer.optimize_model(
        args.model_path,
        'bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=opt_options)
    model = model_optimizer.model
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)

    shapes, lows, highs = parse_dummy_input(model, args.benchmark_nums, args.max_seq_length)

    from lpot.data.datasets.dummy_dataset import DummyDataset
    from lpot.data.dataloaders.onnx_dataloader import ONNXDataLoader
    dummy_dataset = DummyDataset(shapes, low=lows, high=highs, dtype="int64")
    dummy_dataloader = ONNXDataLoader(dummy_dataset)

    def eval_func(model):
        return evaluate_onnx(model, dummy_dataloader)

    if args.benchmark:
        from lpot.benchmark import Benchmark
        benchmark = Benchmark(args.config)
        results = benchmark(model, b_dataloader=dummy_dataloader, b_func=eval_func)

        for mode, result in results.items():
            acc, batch_size, result_list = result
            latency = np.array(result_list).mean() / batch_size

            print('\n{} mode benchmark result:'.format(mode))
            print('Accuracy is {:.3f}'.format(acc))
            print('Batch size = {}'.format(batch_size))
            print('Latency: {:.3f} ms'.format(latency * 1000))
            print('Throughput: {:.3f} items/sec'.format(batch_size * 1./ latency))

    if args.tune:
        from lpot.quantization import Quantization
        quantize = Quantization(args.config)
        q_model = quantize(model, q_dataloader=dummy_dataloader, eval_func=eval_func)
        onnx.save(q_model, args.output_model)
        if args.benchmark:
            from lpot import Benchmark
            benchmark = Benchmark(args.config)
            results = benchmark(model=q_model, b_dataloader=dummy_dataloader, b_func=eval_func)
            for mode, result in results.items():
                acc, batch_size, result_list = result
                latency = np.array(result_list).mean() / batch_size

                print('\n quantized model {} mode benchmark result:'.format(mode))
                print('Accuracy is {:.3f}'.format(acc))
                print('Batch size = {}'.format(batch_size))
                print('Latency: {:.3f} ms'.format(latency * 1000))
                print('Throughput: {:.3f} items/sec'.format(batch_size * 1./ latency))
