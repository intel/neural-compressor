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
import time
import onnxruntime as ort
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def eval_func_onnx(model, dataloader, metric, postprocess=None, batch_size=32, mode='accuracy'):
    metric.reset()
    session = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    ort_inputs = {}
    len_inputs = len(session.get_inputs())
    inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]

    latency_list = []
    for inputs, labels in dataloader:
        if not isinstance(labels, list):
            labels = [labels]
        if len_inputs == 1:
            ort_inputs.update(
                inputs if isinstance(inputs, dict) else {inputs_names[0]: np.array(inputs,dtype=np.uint8)}
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

        start = time.time()
        predictions = session.run(None, ort_inputs)
        end = time.time()

        if postprocess is not None:
            predictions, labels = postprocess((predictions, labels))

        if not hasattr(metric, "compare_label") or \
            (hasattr(metric, "compare_label") and metric.compare_label):
            metric.update(predictions, labels)
        latency_list.append(end-start)
    latency = np.array(latency_list[:]).mean() / batch_size
    if mode == 'performance':
        print("Batch size = {}".format(batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))

    acc = metric.result()
    return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

def eval_func_tf(model, dataloader, metric, postprocess=None, batch_size=32, mode='accuracy'):
    metric.reset()

    from neural_compressor.model import Model
    if isinstance(model, str) or isinstance(model, tf.compat.v1.Graph):
        model = Model(model)
        model.input_tensor_names = ["image_tensor:0"]
        model.output_tensor_names = ["num_detections:0", "detection_boxes:0", \
                                        "detection_scores:0", "detection_classes:0"]
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]

    latency_list = []
    for _, (inputs, labels) in enumerate(dataloader):
        # dataloader should keep the order and len of inputs same with input_tensor
        inputs = np.array([inputs])
        feed_dict = dict(zip(input_tensor, inputs))

        start = time.time()
        predictions = model.sess.run(output_tensor, feed_dict)
        end = time.time()
        metric.update(predictions, labels)
        latency_list.append(end-start)
    latency = np.array(latency_list[:]).mean() / batch_size

    if mode == 'performance':
        print("Batch size = {}".format(batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))

    acc = metric.result()
    return acc

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def __init__(self):
        """Initialization."""
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
                from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
                from neural_compressor.utils.create_obj_from_config import create_dataloader
                calib_dataloader_args = {
                    'dataset': {"COCORecord": {'root':self.args.dataset_location}},
                    'transform': None,
                    'filter': None
                }
                calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
                eval_dataloader_args = {
                    'batch_size': 10,
                    'dataset': {"COCORecord": {'root':self.args.dataset_location}},
                    'transform': {'Resize': {'size': 300}},
                    'filter': None
                }
                eval_dataloader = create_dataloader('tensorflow', eval_dataloader_args)
                conf = PostTrainingQuantConfig(backend='itex', calibration_sampling_size=[10, 50, 100, 200], inputs=['image_tensor'],
                                            outputs=['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'],
                                            accuracy_criterion = AccuracyCriterion(tolerable_loss=0.1))
                from neural_compressor.metric import COCOmAPv2
                output_index_mapping = {'num_detections':0, 'boxes':1, 'scores':2, 'classes':3}
                mAP2 = COCOmAPv2(output_index_mapping=output_index_mapping)
                q_model = quantization.fit(self.args.input_graph, conf=conf, calib_dataloader=calib_dataloader,
                            eval_dataloader=eval_dataloader, eval_metric=mAP2)
                q_model.save("./tf-quant.pb")

                from neural_compressor.config import TF2ONNXConfig
                q_model.input_tensor_names = ["image_tensor"]
                q_model.output_tensor_names = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
                config = TF2ONNXConfig(dtype="int8")
                q_model.export(self.args.output_graph, config)
            else:
                from neural_compressor.model import Model
                from neural_compressor.config import TF2ONNXConfig
                inc_model = Model(self.args.input_graph)
                inc_model.input_tensor_names = ["image_tensor"]
                inc_model.output_tensor_names = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
                config = TF2ONNXConfig(dtype="fp32")
                inc_model.export(self.args.output_graph, config)

        if self.args.benchmark:
            if self.args.input_graph.endswith('.onnx'):
                model = onnx.load(self.args.input_graph)
            else:
                model = self.args.input_graph

            from neural_compressor.utils.create_obj_from_config import create_dataloader
            dataloader_args = {
                    'batch_size': self.args.batch_size,
                    'dataset': {"COCORecord": {'root':self.args.dataset_location}},
                    'transform': {'Resize': {'size': 300}},
                    'filter': None
            }
            dataloader = create_dataloader('tensorflow', dataloader_args)

            from neural_compressor.metric import COCOmAPv2
            output_index_mapping = {'num_detections':0, 'boxes':1, 'scores':2, 'classes':3}
            mAP2 = COCOmAPv2(output_index_mapping=output_index_mapping)

            def eval(model):
                if self.args.input_graph.endswith('.onnx'):
                    return eval_func_onnx(model, dataloader, mAP2,
                                          batch_size=self.args.batch_size, mode='performance')
                else:
                    return eval_func_tf(model, dataloader, mAP2,
                                        batch_size=self.args.batch_size, mode='performance')

            if self.args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1,
                        inputs=['image_tensor'],
                        outputs=['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'])
                fit(model, conf, b_dataloader=dataloader, b_func=eval)
            elif self.args.mode == 'accuracy':
                acc_result = eval(model)
                print("Batch size = %d" % dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)


if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
