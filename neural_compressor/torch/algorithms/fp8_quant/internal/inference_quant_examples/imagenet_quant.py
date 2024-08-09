# Copyright (c) 2024 Intel Corporation
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

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# fp8 additions
import neural_compressor
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import QuantMode, get_hqt_config
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

# data
imgnet_data = "/software/data/pytorch/imagenet/ILSVRC2012/val/"
transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
testset = torchvision.datasets.ImageFolder(imgnet_data, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# Define ResNet-18 model
model = torchvision.models.quantization.resnet18(pretrained=True)
# fp8 additions

config_path = os.getenv("QUANT_CONFIG")
config = FP8Config.from_json_file(config_path)
if config.measure:
    model = prepare(model, config)
elif config.quantize:
    model = convert(model, config)

quant_config = get_hqt_config(model).cfg


# evaluate module
device = "hpu"
model.to(device)
model.eval()


def evaluate():
    accuracy = []
    max_batches = 10 if quant_config["mode"] == QuantMode.MEASURE else 50
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        accurate = 0
        total = 0
        _, predicted = torch.max(output.data, 1)
        # total labels
        total += labels.size(0)
        # Total correct predictions
        accurate += (predicted == labels).sum()
        accuracy_score = 100 * accurate / total
        accuracy.append(accuracy_score)
        if max_batches > 0:
            max_batches -= 1
        else:
            break

    accuracy = [x.item() for x in accuracy]
    print(np.mean(np.array(accuracy)))


with torch.no_grad():

    evaluate()

    # fp8 additions
    if quant_config["mode"] == QuantMode.MEASURE:
        finalize_calibration(model)
