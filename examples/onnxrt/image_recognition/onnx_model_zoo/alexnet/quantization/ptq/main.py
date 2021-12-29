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

import numpy as np
import onnx
import re
import os
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

class dataset:
    def __init__(self, data_path, image_list):
        self.image_list = []
        self.label_list = []
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    continue
                self.image_list.append(src)
                self.label_list.append(int(label))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert('RGB').resize((224, 224))).astype(np.float32)
            image[:, :, 0] -= 123.68
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 103.939
            image[:,:,[0,1,2]] = image[:,:,[2,1,0]]
            image = image.transpose((2, 0, 1))
        return image, label

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="Alexnet fine-tune examples for image classification tasks.",
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
        help="Imagenet data path"
    )
    parser.add_argument(
        '--label_path',
        type=str,
        help="Imagenet label path"
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
        default='performance',
        help="benchmark mode of performance or accuracy"
    )

    args = parser.parse_args()

    model = onnx.load(args.model_path)
    ds = dataset(args.data_path, args.label_path)

    from neural_compressor import options
    options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = common.DataLoader(ds)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.calib_dataloader = common.DataLoader(ds)
        quantize.eval_dataloader = common.DataLoader(ds)
        q_model = quantize()
        q_model.save(args.output_model)
        
