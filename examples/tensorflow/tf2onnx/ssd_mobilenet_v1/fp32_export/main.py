#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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


def eval_func(model, dataloader, metric, postprocess=None):
    metric.reset()
    session = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    # input_names = [i.name for i in sess.get_inputs()]
    # for input_data, label in dataloader:
    #     output = sess.run(None, dict(zip(input_names, [input_data])))
    #     metric.update(output, label)
    # return metric.result()
    ort_inputs = {}
    len_inputs = len(session.get_inputs())
    inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
    for inputs, labels in dataloader:
        if not isinstance(labels, list):
            labels = [labels]
        if len_inputs == 1:
            ort_inputs.update(
                inputs if isinstance(inputs, dict) else {inputs_names[0]: inputs}
            )
        else:
            assert len_inputs == len(inputs), \
                'number of input tensors must align with graph inputs'

            if isinstance(inputs, dict):  # pragma: no cover
                ort_inputs.update(inputs)
            else:
                for i in range(len_inputs):
                    # in case dataloader contains non-array input
                    if not isinstance(inputs[i], np.ndarray):
                        ort_inputs.update({inputs_names[i]: np.array(inputs[i])})
                    else:
                        ort_inputs.update({inputs_names[i]: inputs[i]})

        predictions = session.run(None, ort_inputs)

        if postprocess is not None:
            predictions, labels = postprocess((predictions, labels))

        if not hasattr(metric, "compare_label") or \
            (hasattr(metric, "compare_label") and metric.compare_label):
            metric.update(predictions, labels)
    acc = metric.result()
    return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

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
        arg_parser.add_argument('--dataset_location', dest='dataset_location',
                                 help='location of calibration dataset and evaluate dataset')
        self.args = arg_parser.parse_args()

    def run(self):
        """This is neural_compressor function include tuning and benchmark option."""
        if self.args.export:
            from neural_compressor.model import Model
            from neural_compressor.config import TF2ONNXConfig
            inc_model = Model(self.args.input_graph)
            inc_model.input_tensor_names = ["image_tensor"]
            inc_model.output_tensor_names = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
            config = TF2ONNXConfig(dtype="fp32")
            inc_model.export(self.args.output_graph, config)

        if self.args.benchmark:
            model = onnx.load(self.args.input_graph)

            from neural_compressor.utils.create_obj_from_config import create_dataloader
            dataloader_args = {
                'batch_size': 16,
                'dataset': {"COCORaw": {'root':self.args.dataset_location}},
                'transform': {'Resize': {'size': 300}},
                'filter': None
            }
            dataloader = create_dataloader('onnxrt_integerops', dataloader_args)

            from neural_compressor.metric import COCOmAPv2
            output_index_mapping = {'num_detections':0, 'boxes':1, 'scores':2, 'classes':3}
            mAP2 = COCOmAPv2(output_index_mapping=output_index_mapping)
            def eval(onnx_model):
                return eval_func(onnx_model, dataloader, mAP2)

            if self.args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=7)
                fit(model, conf, b_dataloader=dataloader)
            elif self.args.mode == 'accuracy':
                acc_result = eval(model)
                print("Batch size = %d" % dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
