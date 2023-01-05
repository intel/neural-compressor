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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def eval_func(model, dataloader, metric, postprocess):
    metric.reset()
    sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    input_names = [i.name for i in sess.get_inputs()]
    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))
        output, label = postprocess((output, label))
        metric.update(output[1], label)
    return metric.result()


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
            config = TF2ONNXConfig(dtype="fp32", inputs_as_nchw="input_tensor:0")
            inc_model.export(self.args.output_graph, config)

        if self.args.benchmark:
            model = onnx.load(self.args.input_graph)
            data_path = os.path.join(self.args.dataset_location, 'ILSVRC2012_img_val')
            label_path = os.path.join(self.args.dataset_location, 'val.txt')

            from neural_compressor.utils.create_obj_from_config import create_dataloader
            dataloader_args = {
                'batch_size': 32,
                'dataset': {"ImagenetRaw": {'data_path':data_path, 'image_list':label_path}},
                'transform': {'ResizeWithAspectRatio': {'height': 224, 'width': 224},
                              'CenterCrop': {'size': 224},
                              'Normalize': {'mean': [123.68, 116.78, 103.94]},
                              'Cast': {'dtype': 'float32'},
                              'Transpose': {'perm': [2, 0, 1]}},
                'filter': None
            }
            dataloader = create_dataloader('onnxrt_integerops', dataloader_args)

            from neural_compressor.metric import GeneralTopK
            top1 = GeneralTopK(k=1)
            from neural_compressor.data.transforms.imagenet_transform import LabelShift
            postprocess = LabelShift(label_shift=-1)
            def eval(onnx_model):
                return eval_func(onnx_model, dataloader, top1, postprocess)

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
