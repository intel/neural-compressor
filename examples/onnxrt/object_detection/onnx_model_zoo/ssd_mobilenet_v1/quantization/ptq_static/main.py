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
import onnxruntime as ort
import numpy as np

from data_utils import COCORawDataloader, COCORawDataset, COCOmAPv2
from data_utils import ComposeTransform, ResizeTransform, LabelBalanceCOCORawFilter

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
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
    '--data_path',
    type=str,
    help="path to dataset"
)
parser.add_argument(
    '--label_path',
    type=str,
    default='label_map.yaml',
    help="Path of label map yaml file"
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
parser.add_argument(
    '--quant_format',
    type=str,
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help="quantization format"
)
args = parser.parse_args()

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    filter = LabelBalanceCOCORawFilter()
    eval_dataset = COCORawDataset(args.data_path)
    calib_dataset = COCORawDataset(args.data_path, filter=filter)
    eval_dataloader = COCORawDataloader(eval_dataset, batch_size=args.batch_size)
    calib_dataloader = COCORawDataloader(calib_dataset, 1)
    metric = COCOmAPv2(anno_path=args.label_path, output_index_mapping={'boxes': 0,
                                                                         'classes': 1,
                                                                         'scores': 2,
                                                                         'num_detections': 3})

    def eval_func(model):
        metric.reset()
        session = ort.InferenceSession(model.SerializeToString(), 
                                       providers=ort.get_available_providers())
        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        for idx, (inputs, labels) in enumerate(eval_dataloader):
            if not isinstance(labels, list):
                labels = [labels]
            if len_inputs == 1:
                ort_inputs.update(
                    inputs if isinstance(inputs, dict) else {inputs_names[0]: inputs}
                )
            else:
                assert len_inputs == len(inputs), 'number of input tensors must align with graph inputs'
                if isinstance(inputs, dict):
                    ort_inputs.update(inputs)
                else:
                    for i in range(len_inputs):
                        if not isinstance(inputs[i], np.ndarray):
                            ort_inputs.update({inputs_names[i]: np.array(inputs[i])})
                        else:
                            ort_inputs.update({inputs_names[i]: inputs[i]})
            predictions = session.run(None, ort_inputs)
            metric.update(predictions, labels)
        return metric.result()

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(iteration=100,
                                   cores_per_instance=4,
                                   num_of_instance=1)
            fit(model, conf, b_dataloader=eval_dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval_func(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        config = PostTrainingQuantConfig(approach='static', 
                                         quant_format=args.quant_format)
        q_model = quantization.fit(model, config, calib_dataloader=calib_dataloader, eval_func=eval_func)
        q_model.save(args.output_model)