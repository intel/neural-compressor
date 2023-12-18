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
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)
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
    '--diagnose',
    dest='diagnose',
    action='store_true',
    help='use Neural Insights to diagnose tuning and benchmark.',
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
    default=16,
    help="quantization format"
)
parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    choices=['cpu', 'npu'],
)
args = parser.parse_args()
backend = 'onnxrt_dml_ep' if args.device == 'npu' else 'default'

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    transform = ComposeTransform([ResizeTransform(size=300)])
    filter = LabelBalanceCOCORawFilter()
    eval_dataset = COCORawDataset(args.data_path, transform=transform)
    calib_dataset = COCORawDataset(args.data_path, transform=transform, filter=filter)
    eval_dataloader = COCORawDataloader(eval_dataset, batch_size=args.batch_size)
    calib_dataloader = COCORawDataloader(calib_dataset, 1)
    metric = COCOmAPv2(output_index_mapping={'num_detections': 0,
                                             'boxes': 1,
                                             'scores': 2,
                                             'classes': 3})

    def eval_func(model):
        metric.reset()
        provider = 'DmlExecutionProvider' if backend == 'onnxrt_dml_ep' else 'CPUExecutionProvider'
        session = ort.InferenceSession(model.SerializeToString(), providers=[provider])
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
        if args.diagnose and args.mode != "performance":
            print("[ WARNING ] Diagnosis works only with performance benchmark.")
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(
                iteration=100,
                cores_per_instance=4,
                num_of_instance=1,
                diagnosis=args.diagnose,
                device=args.device,
                backend=backend,
            )
            fit(model, conf, b_dataloader=eval_dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval_func(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
        accuracy_criterion = AccuracyCriterion()
        accuracy_criterion.absolute = 0.01
        config = PostTrainingQuantConfig(
            approach='static',
            accuracy_criterion=accuracy_criterion,
            quant_format=args.quant_format,
            calibration_sampling_size=[50],
            diagnosis=args.diagnose,
            device=args.device,
            backend=backend,
        )
        q_model = quantization.fit(model, config, calib_dataloader=calib_dataloader, eval_func=eval_func)
        q_model.save(args.output_model)
