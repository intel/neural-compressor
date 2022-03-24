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

import argparse
import logging
from pathlib import Path

import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from neural_compressor.adaptor.mxnet_utils.util import check_mx_version


def main():
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description='Generate a calibrated quantized model from a FP32 model with oneDNN support')
    parser.add_argument('--model_name', type=str, default='resnet50_v1',
                        help='''model to be quantized. If no-pretrained is set then model must be 
                            provided to `model` directory in the same path as this python script.
                            Grouped supported models at the time of commit:
                            - alexnet
                            - densenet121, densenet161
                            - densenet169, densenet201
                            - inceptionv3
                            - mobilenet0.25, mobilenet0.5, mobilenet0.75, mobilenet1.0,
                            - mobilenetv2_0.25, mobilenetv2_0.5, mobilenetv2_0.75, mobilenetv2_1.0
                            - resnet101_v1, resnet152_v1, resnet18_v1, resnet34_v1, resnet50_v1
                            - resnet101_v2, resnet152_v2, resnet18_v2, resnet34_v2, resnet50_v2
                            - squeezenet1.0, squeezenet1.1
                            - vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
                            ''')
    parser.add_argument('--model_path', type=str, default='./model',
                        help='directory to put models, default is ./model')
    args = parser.parse_args()
    model_name = args.model_name
    model_path = Path(args.model_path).resolve()
    image_shape = 299 if model_name == 'inceptionv3' else 224
    data_shape = (1, 3, image_shape, image_shape)

    model_path.mkdir(parents=True, exist_ok=True)

    net = get_model(name=model_name, classes=1000, pretrained=True)
    net.hybridize()
    # forward pass to build the graph
    if check_mx_version('2.0.0'):
        net(mx.np.zeros(data_shape))
    else:
        net(mx.nd.zeros(data_shape))
    net.export(str(model_path/model_name))


if __name__ == '__main__':
    main()
