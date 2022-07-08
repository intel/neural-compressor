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

import argparse
import logging
import numpy as np
import onnx
import pandas as pd

from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--data_path',
    type=str,
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
args = parser.parse_args()

class Dataloader:
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df[df['Usage']=='PublicTest']
        images = [np.reshape(np.fromstring(image, dtype=np.uint8, sep=' '), (48, 48)) for image in df['pixels']]
        labels = np.array(list(map(int, df['emotion'])))
        self.batch_size = 1
        self.data = [(self.preprocess(image), label) for image, label in zip(images, labels)]
            
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield item

    def preprocess(self, image):
        input_shape = (1, 1, 64, 64)
        img = Image.fromarray(image)
        img = img.resize((64, 64), Image.ANTIALIAS)
        img_data = np.array(img)
        img_data = np.resize(img_data, input_shape)
        return img_data.astype('float32')

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    dataloader  = Dataloader(args.data_path)
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        if args.mode == 'accuracy':
            evaluator = Benchmark(args.config)
            evaluator.model = common.Model(model)
            evaluator.b_dataloader = dataloader
            evaluator(args.mode)
        else:
            evaluator = Benchmark(args.config)
            evaluator.model = common.Model(model)
            evaluator(args.mode)

    if args.tune:
        from neural_compressor import options
        from neural_compressor.experimental import Quantization, common
        options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.eval_dataloader = dataloader
        quantize.calib_dataloader = dataloader
        q_model = quantize()
        q_model.save(args.output_model)