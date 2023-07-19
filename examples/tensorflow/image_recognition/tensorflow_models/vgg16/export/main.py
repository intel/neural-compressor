#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
#

import time
import onnx
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from argparse import ArgumentParser
from neural_compressor.data import LabelShift
from neural_compressor.utils.create_obj_from_config import create_dataloader

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('-g', "--input-graph",
                        help='Specify the input graph for the transform tool',
                        dest='input_graph')
arg_parser.add_argument("--output-graph",
                        help='Specify tune result model save dir',
                        dest='output_graph')
arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')
arg_parser.add_argument('--mode', dest='mode', default='performance', help='benchmark mode')
arg_parser.add_argument('--export', dest='export', action='store_true', help='use neural_compressor to export.')
arg_parser.add_argument('--dataset_location', dest='dataset_location',
                            help='location of calibration dataset and evaluate dataset')
arg_parser.add_argument('--batch_size', type=int, default=32, dest='batch_size', help='batch_size of benchmark')
arg_parser.add_argument('--dtype', dest='dtype', default='fp32', help='the data type of export')
arg_parser.add_argument('--quant_format', dest='quant_format', default='qdq', help='the quant format of export')
args = arg_parser.parse_args()

def eval_func_onnx(model, dataloader, metric, postprocess=None):
    metric.reset()
    sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    input_names = [i.name for i in sess.get_inputs()]

    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))

        output, label = postprocess((output, label))
        metric.update(output, label)

    acc = metric.result()
    return acc

def eval_func_tf(model, eval_dataloader, metric, postprocess=None):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model import Model
    model = Model(model)
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1
    if args.benchmark and args.mode == 'performance':
        iteration = args.iters

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            inputs = np.array([inputs])
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()
            if postprocess:
                predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == iteration:
                break
        latency = np.array(latency_list).mean() / args.batch_size
        return latency

    latency = eval_func(eval_dataloader)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def run(self):
        """This is neural_compressor function include tuning, export and benchmark option."""
        if args.quant_format != 'qdq':
            raise ValueError("Only support tensorflow export to ONNX for QDQ format, "
                "please make sure input the correct quant_format.")

        postprocess = LabelShift(label_shift=1)

        if args.export:
            if args.dtype == 'int8':
                from neural_compressor import quantization
                from neural_compressor.config import PostTrainingQuantConfig
                calib_dataloader_args = {
                    'batch_size': 10,
                    'dataset': {"ImageRecord": {'root':args.dataset_location}},
                    'transform': {'ResizeCropImagenet':
                        {'height': 224, 'width': 224, 'mean_value': [123.68, 116.78, 103.94]}},
                    'filter': None
                }
                calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
                eval_dataloader_args = {
                    'batch_size': 32,
                    'dataset': {"ImageRecord": {'root':args.dataset_location}},
                    'transform': {'ResizeCropImagenet':
                            {'height': 224, 'width': 224, 'mean_value': [123.68, 116.78, 103.94]}},
                    'filter': None
                }
                eval_dataloader = create_dataloader('tensorflow', eval_dataloader_args)
                conf = PostTrainingQuantConfig(backend='itex', calibration_sampling_size=[50, 100],
                                            outputs=['softmax_tensor'])
                from neural_compressor import METRICS
                metrics = METRICS('tensorflow')
                top1 = metrics['topk']()
                def eval(model):
                    return eval_func_tf(model, eval_dataloader, top1, postprocess)
                q_model = quantization.fit(args.input_graph, conf=conf, calib_dataloader=calib_dataloader,
                            eval_func=eval)
                q_model.save("./tf-quant.pb")
                from neural_compressor.config import TF2ONNXConfig
                config = TF2ONNXConfig(dtype=args.dtype)
                q_model.export(args.output_graph, config)
            else:
                from neural_compressor.model import Model
                from neural_compressor.config import TF2ONNXConfig
                inc_model = Model(args.input_graph)
                config = TF2ONNXConfig(dtype=args.dtype)
                inc_model.export(args.output_graph, config)

        if args.benchmark:
            if args.input_graph.endswith('.onnx'):
                model = onnx.load(args.input_graph)
            else:
                model = args.input_graph
            eval_dataloader_args = {
                'batch_size': args.batch_size,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet':
                     {'height': 224, 'width': 224, 'mean_value': [123.68, 116.78, 103.94]}},
                'filter': None
            }
            eval_dataloader = create_dataloader('tensorflow', eval_dataloader_args)
            def eval(model):
                if isinstance(model, str):
                    from neural_compressor import METRICS
                    metrics = METRICS('tensorflow')
                    top1 = metrics['topk']()
                    return eval_func_tf(model, eval_dataloader, top1, postprocess)
                else:
                    from neural_compressor import METRICS
                    metrics = METRICS('onnxrt_qlinearops')
                    top1 = metrics['topk']()
                    return eval_func_onnx(model, eval_dataloader, top1, postprocess)

            if args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
                fit(model, conf, b_dataloader=eval_dataloader)
            elif args.mode == 'accuracy':
                acc_result = eval(model)
                print("Batch size = %d" % eval_dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
