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
# ==============================================================================
"""Dataset used for style transfer task on multiple framework backends."""

import glob
import os

import numpy as np

from .dataset import Dataset, dataset_registry


@dataset_registry(
    dataset_type="style_transfer",
    framework="tensorflow, \
                  tensorflow_itex",
    dataset_format="",
)
class StyleTransferDataset(Dataset):  # pragma: no cover
    """Dataset used for style transfer task on tensorflow/inteltensorflow/tensorflow_itex backend.

    This Dataset is to construct a dataset from two specific image holders representing
        content image folder and style image folder.
    """

    def __init__(
        self,
        content_folder,
        style_folder,
        crop_ratio=0.1,
        resize_shape=(256, 256),
        image_format="jpg",
        transform=None,
        filter=None,
    ):
        """Initialize `StyleTransferDataset` class.

        Args:
            content_folder (str): Root directory of content images.
            style_folder (str): Root directory of style images.
            crop_ratio (float, default=0.1): Cropped ratio to each side.
            resize_shape (tuple, default=(256, 256)): Target size of image.
            image_format (str, default='jpg'): Target image format.
            transform (transform object, default=None): Transform to process input data.
            filter (Filter objects, default=None): Filter out examples according to specific conditions.
        """
        self.transform = transform
        self.content_folder = content_folder
        self.style_folder = style_folder
        self.resize_shape = resize_shape
        self.crop_ratio = crop_ratio
        self.content_images = glob.glob(os.path.join(content_folder, "*" + image_format))
        self.style_images = glob.glob(os.path.join(style_folder, "*" + image_format))
        self.image_list = []
        for content in self.content_images:
            for style in self.style_images:
                self.image_list.append((content, style))

    def __len__(self):
        """Return the length of dataset."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        from PIL import Image

        content_image, style_image = self.image_list[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        width, height = style_image.size
        crop_ratio = self.crop_ratio
        crop_box = (crop_ratio * height, crop_ratio * width, (1 - crop_ratio) * height, (1 - crop_ratio) * width)
        content_image = np.asarray(content_image.resize(self.resize_shape))
        style_image = np.asarray(style_image.resize(self.resize_shape))
        if content_image.max() > 1.0:
            content_image = content_image / 255.0
        if style_image.max() > 1.0:
            style_image = style_image / 255.0

        return (content_image, style_image), 0
