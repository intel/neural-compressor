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

from argparse import ArgumentParser
import tensorflow as tf
import onnx
import os
import onnxruntime as ort
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def eval_func_onnx(model, dataloader, metric, postprocess=None):
    metric.reset()
    sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    input_names = [i.name for i in sess.get_inputs()]
    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))
        if postprocess:
            output, label = postprocess((output, label))
        metric.update(output, label)
    return metric.result()

def eval_func_tf(model, dataloader, metric, postprocess=None):
    from neural_compressor.model import Model
    model = Model(model)
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]

    for _, (inputs, labels) in enumerate(dataloader):
        # dataloader should keep the order and len of inputs same with input_tensor
        inputs = np.array([inputs])
        feed_dict = dict(zip(input_tensor, inputs))
        predictions = model.sess.run(output_tensor, feed_dict)
        metric.update(predictions, labels)
    acc = metric.result()
    return acc

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def __init__(self):
        """Initilization."""
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
        arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use neural_compressor to tune.')
        arg_parser.add_argument('--dataset_location', dest='dataset_location',
                                 help='location of calibration dataset and evaluate dataset')
        arg_parser.add_argument('--batch_size', type=int, default=32, dest='batch_size', help='batch_size of benchmark')
        arg_parser.add_argument('--dtype', dest='dtype', default='fp32', help='the data type of export')
        arg_parser.add_argument('--quant_format', dest='quant_format', default='qdq', help='the quant format of export')
        self.args = arg_parser.parse_args()

    def run(self):
        """This is neural_compressor function include tuning, export and benchmark option."""
        if self.args.quant_format != 'qdq':
            raise ValueError("Only support tensorflow export to ONNX for QDQ format, "
                "please make sure input the correct quant_format.")

        if self.args.export:
            if self.args.dtype == 'int8':
                from neural_compressor import quantization
                from neural_compressor.config import PostTrainingQuantConfig
                from neural_compressor.utils.create_obj_from_config import create_dataloader
                dataloader_args = {
                    'batch_size': 10,
                    'dataset': {"ImageRecord": {'root': self.args.dataset_location}},
                    'transform': {'BilinearImagenet':
                        {'height': 224, 'width': 224}},
                    'filter': None
                }
                dataloader = create_dataloader('tensorflow', dataloader_args)
                conf = PostTrainingQuantConfig(backend='itex', calibration_sampling_size=[50, 100])
                from neural_compressor import Metric
                top1 = Metric(name="topk", k=1)
                q_model = quantization.fit(self.args.input_graph, conf=conf, calib_dataloader=dataloader,
                            eval_dataloader=dataloader, eval_metric=top1)
                q_model.save("./tf-quant.pb")
                from neural_compressor.config import TF2ONNXConfig
                config = TF2ONNXConfig(dtype=self.args.dtype, input_names='input[-1,224,224,3]')
                q_model.export(self.args.output_graph, config)
            else:
                from neural_compressor.model import Model
                from neural_compressor.config import TF2ONNXConfig
                inc_model = Model(self.args.input_graph)
                config = TF2ONNXConfig(dtype="fp32", input_names='input[-1,224,224,3]')
                inc_model.export(self.args.output_graph, config)

        if self.args.benchmark:
            if self.args.input_graph.endswith('.onnx'):
                model = onnx.load(self.args.input_graph)
            else:
                model = self.args.input_graph

            from neural_compressor.utils.create_obj_from_config import create_dataloader
            dataloader_args = {
                'batch_size': self.args.batch_size,
                'dataset': {"ImageRecord": {'root': self.args.dataset_location}},
                'transform': {'BilinearImagenet': {'height': 224, 'width': 224}},
                'filter': None
            }
            dataloader = create_dataloader('tensorflow', dataloader_args)
            from neural_compressor import METRICS
            metrics = METRICS('tensorflow')
            top1 = metrics['topk']()
            def eval(model):
                if isinstance(model, str):
                    return eval_func_tf(model, dataloader, top1)
                else:
                    return eval_func_onnx(model, dataloader, top1)

            if self.args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
                fit(self.args.input_graph, conf, b_dataloader=dataloader)
            elif self.args.mode == 'accuracy':
                acc_result = eval(model)
                print("Batch size = %d" % dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
