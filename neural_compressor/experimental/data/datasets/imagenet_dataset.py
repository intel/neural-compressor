#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import re
import numpy as np
from PIL import Image
from neural_compressor.utils.utility import LazyImport
from .dataset import dataset_registry, IterableDataset, Dataset
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')
torch = LazyImport('torch')

@dataset_registry(dataset_type="ImagenetRaw", framework="onnxrt_qlinearops, \
                    onnxrt_integerops", dataset_format='')
class ImagenetRaw(Dataset):
    """Configuration for Imagenet Raw dataset.

    Please arrange data in this way:  
        data_path/img1.jpg  
        data_path/img2.jpg  
        ...  
        data_path/imgx.jpg  
    dataset will read name and label of each image from image_list file, 
    if user set image_list to None, it will read from data_path/val_map.txt automatically.

    Args: data_path (str): Root directory of dataset.
          image_list (str): data file, record image_names and their labels.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according to 
                                                 specific conditions
    """
    def __init__(self, data_path, image_list, transform=None, filter=None):
        self.image_list = []
        self.label_list = []
        self.data_path = data_path
        self.transform = transform
        not_found = 0
        if image_list is None:
            # by default look for val.txt
            image_list = os.path.join(data_path, "val.txt")

        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                self.image_list.append(src)
                self.label_list.append(int(label))

        if not self.image_list:
            raise ValueError("no images in image list found")
        if not_found > 0:
            print("reduced image list, %d images not found", not_found)

    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert('RGB'))
            if self.transform is not None:
                image, label = self.transform((image, label))
            return (image, label)

    def __len__(self):
        return len(self.image_list)

@dataset_registry(dataset_type="ImagenetRaw", framework="pytorch", dataset_format='')
class PytorchImagenetRaw(ImagenetRaw):
    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if self.transform is not None:
                image, label = self.transform((image, label))
            image = np.array(image)
            return (image, label)

@dataset_registry(dataset_type="ImagenetRaw", framework="mxnet", dataset_format='')
class MXNetImagenetRaw(ImagenetRaw):
    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        image = mx.image.imread(image_path)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)

@dataset_registry(dataset_type="ImagenetRaw", framework="tensorflow", 
                    dataset_format='')
class TensorflowImagenetRaw(ImagenetRaw):
    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert('RGB'))
            if self.transform is not None:
                image, label = self.transform((image, label))
            if type(image).__name__ == 'Tensor':
                with tf.compat.v1.Session() as sess:
                    image = sess.run(image)
            elif type(image).__name__ == 'EagerTensor':
                image = image.numpy()
            return (image, label)
