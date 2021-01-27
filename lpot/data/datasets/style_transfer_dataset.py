#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import glob
from .dataset import dataset_registry, Dataset


@dataset_registry(dataset_type="style_transfer", framework="tensorflow", dataset_format='')
class StyleTransferDataset(Dataset):
    """Dataset used for style transfer task.
       This Dataset is to construct a dataset from two specific image holders representing
       content image folder and style image folder.

    """

    def __init__(self, content_folder, style_folder, crop_ratio=0.1,
                 resize_shape=(256, 256), image_format='jpg', transform=None, filter=None):

        self.transform = transform
        self.content_folder = content_folder
        self.style_folder = style_folder
        self.resize_shape = resize_shape
        self.crop_ratio = crop_ratio
        self.content_images = glob.glob(os.path.join(content_folder, '*' + image_format))
        self.style_images = glob.glob(os.path.join(style_folder, '*' + image_format))
        self.image_list = []
        for content in self.content_images:
            for style in self.style_images:
                self.image_list.append((content, style))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        from PIL import Image
        content_image, style_image = self.image_list[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        width, height = style_image.size
        crop_ratio = self.crop_ratio
        crop_box = (
            crop_ratio * height,
            crop_ratio * width,
            (1 - crop_ratio) * height,
            (1 - crop_ratio) * width)
        content_image = np.asarray(content_image.resize(self.resize_shape))
        style_image = np.asarray(style_image.resize(self.resize_shape))
        if content_image.max() > 1.0:
            content_image = content_image / 255.
        if style_image.max() > 1.0:
            style_image = style_image / 255.

        return (content_image, style_image), 0
