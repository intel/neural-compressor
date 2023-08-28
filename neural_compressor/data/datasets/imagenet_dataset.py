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
"""Dataset for ImageNet data generation on multiple framework backends."""

import os
import re

import numpy as np
from PIL import Image

from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

from .dataset import Dataset, IterableDataset, dataset_registry

tf = LazyImport("tensorflow")
mx = LazyImport("mxnet")
torch = LazyImport("torch")


@dataset_registry(
    dataset_type="ImagenetRaw",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class ImagenetRaw(Dataset):  # pragma: no cover
    """Configuration for ImageNet raw dataset.

    Please arrange data in this way:
        data_path/img1.jpg
        data_path/img2.jpg
        ...
        data_path/imgx.jpg
    dataset will read name and label of each image from image_list file,
    if user set image_list to None, it will read from data_path/val_map.txt automatically.
    """

    def __init__(self, data_path, image_list, transform=None, filter=None):
        """Initialize `ImagenetRaw` class.

        Args:
            data_path (str): Root directory of dataset.
            image_list (str): Data file, record image_names and their labels.
            transform (transform object, default=None): Transform to process input data.
            filter (Filter objects, default=None): Filter out examples according to specific conditions.
        """
        self.image_list = []
        self.label_list = []
        self.data_path = data_path
        self.transform = transform
        not_found = 0
        if image_list is None:
            # by default look for val.txt
            image_list = os.path.join(data_path, "val.txt")

        with open(image_list, "r") as f:
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
        """Return the item of dataset according to the given index."""
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert("RGB"))
            if self.transform is not None:
                image, label = self.transform((image, label))
            return (image, label)

    def __len__(self):
        """Return the length of dataset."""
        return len(self.image_list)


@dataset_registry(dataset_type="ImagenetRaw", framework="pytorch", dataset_format="")
class PytorchImagenetRaw(ImagenetRaw):  # pragma: no cover
    """Dataset for ImageNet data generation on pytorch backend."""

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image, label = self.transform((image, label))
            image = np.array(image)
            return (image, label)


@dataset_registry(dataset_type="ImagenetRaw", framework="mxnet", dataset_format="")
class MXNetImagenetRaw(ImagenetRaw):  # pragma: no cover
    """Dataset for ImageNet data generation on mxnet backend."""

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        image_path, label = self.image_list[index], self.label_list[index]
        image = mx.image.imread(image_path)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(
    dataset_type="ImagenetRaw",
    framework="tensorflow, \
                  tensorflow_itex",
    dataset_format="",
)
class TensorflowImagenetRaw(ImagenetRaw):  # pragma: no cover
    """Dataset for ImageNet data generation on tensorflow/inteltensorflow/tensorflow_itex backend."""

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert("RGB"))
            if self.transform is not None:
                image, label = self.transform((image, label))
            if type(image).__name__ == "Tensor":
                with tf.compat.v1.Session() as sess:
                    image = sess.run(image)
            elif type(image).__name__ == "EagerTensor":
                image = image.numpy()
            return (image, label)


@dataset_registry(dataset_type="Imagenet", framework="tensorflow", dataset_format="")
class TensorflowImagenetDataset(IterableDataset):  # pragma: no cover
    """Configuration for Imagenet dataset."""

    def __new__(cls, root, subset="validation", num_cores=28, transform=None, filter=None):
        """New a imagenet dataset for tensorflow."""
        assert subset in ("validation", "train"), "only support subset (validation, train)"
        logger.warning("This api is going to be deprecated, " "please use ImageRecord instead.")

        from tensorflow.python.platform import gfile

        glob_pattern = os.path.join(root, "%s-*-of-*" % subset)
        file_names = gfile.Glob(glob_pattern)
        if not file_names:
            raise ValueError("Found no files in --root matching: {}".format(glob_pattern))

        from tensorflow.python.data.experimental import parallel_interleave

        from neural_compressor.data.transforms.imagenet_transform import ParseDecodeImagenet

        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_cores))

        if transform is not None:
            transform.transform_list.insert(0, ParseDecodeImagenet())
        else:
            transform = ParseDecodeImagenet()
        ds = ds.map(transform, num_parallel_calls=None)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds


@dataset_registry(
    dataset_type="Imagenet",
    framework="onnxrt_qlinearops, \
                   onnxrt_integerops",
    dataset_format="",
)
class ONNXRTImagenetDataset(Dataset):  # pragma: no cover
    """Configuration for Imagenet dataset."""

    def __init__(self, root, subset="val", num_cores=28, transform=None, filter=None):
        """Initialize `ONNXRTImagenetDataset` class."""
        self.val_dir = os.path.join(root, subset)
        assert os.path.exists(self.val_dir), (
            "find no val dir in {}".format(root) + "please make sure there are train/val subfolders"
        )
        import glob

        logger.warning("This api is going to be deprecated, " "please use ImageRecord instead.")

        self.transform = transform
        self.image_list = []
        files = glob.glob(os.path.join(self.val_dir, "*"))
        files.sort()
        for idx, file in enumerate(files):
            imgs = glob.glob(os.path.join(file, "*"))
            for img in imgs:
                self.image_list.append((img, idx))

    def __len__(self):
        """Return the number of images."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        from PIL import Image

        sample = self.image_list[index]
        image = Image.open(sample[0])
        if self.transform is not None:
            image, label = self.transform((image, sample[1]))
            return (image, label)
