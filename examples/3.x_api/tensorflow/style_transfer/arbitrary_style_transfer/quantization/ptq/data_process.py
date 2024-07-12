#
# -*- coding: utf-8 -*-
#
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
#
import os
import glob
import collections

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from neural_compressor.common import logger
from neural_compressor.tensorflow.utils.data import default_collate


class StyleTransferDataset(object):
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


class ComposeTransform(object):
    """Composes several transforms together.

    Args:
        transform_list (list of Transform objects):  list of transforms to compose

    Returns:
        sample (tuple): tuple of processed image and label
    """

    def __init__(self, transform_list):
        """Initialize `ComposeTransform` class."""
        self.transform_list = transform_list

    def __call__(self, sample):
        """Call transforms in transform_list."""
        for transform in self.transform_list:
            sample = transform(sample)
        return sample

class ParseDecodeVocTransform():
    """Parse features in Example proto.

    Returns:
        tuple of parsed image and labels
    """

    def __call__(self, sample):
        """Parse decode voc."""

        # Currently only supports jpeg and png.
        # Need to use this logic because the shape is not known for
        # tf.image.decode_image and we rely on this info to
        # extend label if necessary.
        def _decode_image(content, channels):
            """Decode the image with content."""
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels),
            )

        features = {
            "image/encoded": tf.compat.v1.FixedLenFeature((), tf.string, default_value=""),
            "image/filename": tf.compat.v1.FixedLenFeature((), tf.string, default_value=""),
            "image/format": tf.compat.v1.FixedLenFeature((), tf.string, default_value="jpeg"),
            "image/height": tf.compat.v1.FixedLenFeature((), tf.int64, default_value=0),
            "image/width": tf.compat.v1.FixedLenFeature((), tf.int64, default_value=0),
            "image/segmentation/class/encoded": tf.compat.v1.FixedLenFeature((), tf.string, default_value=""),
            "image/segmentation/class/format": tf.compat.v1.FixedLenFeature((), tf.string, default_value="png"),
        }

        parsed_features = tf.compat.v1.parse_single_example(sample, features)

        image = _decode_image(parsed_features["image/encoded"], channels=3)

        label = None
        label = _decode_image(parsed_features["image/segmentation/class/encoded"], channels=1)

        sample = {
            "image": image,
        }

        label.set_shape([None, None, 1])

        sample["labels_class"] = label

        return sample["image"], sample["labels_class"]


class BaseMetric(object):
    """The base class of Metric."""

    def __init__(self, metric, single_output=False, hvd=None):
        """Initialize the basic metric.

        Args:
            metric: The metric class.
            single_output: Whether the output is single or not, defaults to False.
            hvd: The Horovod class for distributed training, defaults to None.
        """
        self._metric_cls = metric
        self._single_output = single_output
        self._hvd = hvd

    def __call__(self, *args, **kwargs):
        """Evaluate the model predictions, and the reference.

        Returns:
            The class itself.
        """
        self._metric = self._metric_cls(*args, **kwargs)
        return self

    @abstractmethod
    def update(self, preds, labels=None, sample_weight=None):
        """Update the state that need to be evaluated.

        Args:
            preds: The prediction result.
            labels: The reference. Defaults to None.
            sample_weight: The sampling weight. Defaults to None.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Clear the predictions and labels.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def result(self):
        """Evaluate the difference between predictions and labels.

        Raises:
            NotImplementedError: The method should be implemented by subclass.
        """
        raise NotImplementedError

    @property
    def metric(self):
        """Return its metric class.

        Returns:
            The metric class.
        """
        return self._metric_cls

    @property
    def hvd(self):
        """Return its hvd class.

        Returns:
            The hvd class.
        """
        return self._hvd

    @hvd.setter
    def hvd(self, hvd):
        """Set its hvd.

        Args:
            hvd: The Horovod class for distributed training.
        """
        self._hvd = hvd


class TopKMetric(BaseMetric):
    """Compute Top-k Accuracy classification score for Tensorflow model.

    This metric computes the number of times where the correct label is among
    the top k labels predicted.

    Attributes:
        k (int): The number of most likely outcomes considered to find the correct label.
        num_correct: The number of predictions that were correct classified.
        num_sample: The total number of predictions.
    """

    def __init__(self, k=1):
        """Initialize the k, number of samples and correct predictions.

        Args:
            k: The number of most likely outcomes considered to find the correct label.
        """
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        preds, labels = TopKMetric._topk_shape_validate(preds, labels)

        labels = labels.reshape([len(labels)])
        with tf.Graph().as_default() as acc_graph:
            topk = tf.nn.in_top_k(
                predictions=tf.constant(preds, dtype=tf.float32), targets=tf.constant(labels, dtype=tf.int32), k=self.k
            )
            fp32_topk = tf.cast(topk, tf.float32)
            correct_tensor = tf.reduce_sum(input_tensor=fp32_topk)

            with tf.compat.v1.Session() as acc_sess:
                correct = acc_sess.run(correct_tensor)

        self.num_sample += len(labels)
        self.num_correct += correct

    def reset(self):
        """Reset the number of samples and correct predictions."""
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        """Compute the top-k score.

        Returns:
            The top-k score.
        """
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        elif getattr(self, "_hvd", None) is not None:  # pragma: no cover
            allgather_num_correct = sum(self._hvd.allgather_object(self.num_correct))
            allgather_num_sample = sum(self._hvd.allgather_object(self.num_sample))
            return allgather_num_correct / allgather_num_sample
        return self.num_correct / self.num_sample

    @staticmethod
    def _topk_shape_validate(preds, labels):
        # preds shape can be Nxclass_num or class_num(N=1 by default)
        # it's more suitable for 'Accuracy' with preds shape Nx1(or 1) output from argmax
        if isinstance(preds, int):
            preds = [preds]
            preds = np.array(preds)
        elif isinstance(preds, np.ndarray):
            preds = np.array(preds)
        elif isinstance(preds, list):
            preds = np.array(preds)
            preds = preds.reshape((-1, preds.shape[-1]))

        # consider labels just int value 1x1
        if isinstance(labels, int):
            labels = [labels]
            labels = np.array(labels)
        elif isinstance(labels, tuple):
            labels = np.array([labels])
            labels = labels.reshape((labels.shape[-1], -1))
        elif isinstance(labels, list):
            if isinstance(labels[0], int):
                labels = np.array(labels)
                labels = labels.reshape((labels.shape[0], 1))
            elif isinstance(labels[0], tuple):
                labels = np.array(labels)
                labels = labels.reshape((labels.shape[-1], -1))
            else:
                labels = np.array(labels)
        # labels most have 2 axis, 2 cases: N(or Nx1 sparse) or Nxclass_num(one-hot)
        # only support 2 dimension one-shot labels
        # or 1 dimension one-hot class_num will confuse with N

        if len(preds.shape) == 1:
            N = 1
            class_num = preds.shape[0]
            preds = preds.reshape([-1, class_num])
        elif len(preds.shape) >= 2:
            N = preds.shape[0]
            preds = preds.reshape([N, -1])
            class_num = preds.shape[1]

        label_N = labels.shape[0]
        assert label_N == N, "labels batch size should same with preds"
        labels = labels.reshape([N, -1])
        # one-hot labels will have 2 dimension not equal 1
        if labels.shape[1] != 1:
            labels = labels.argsort()[..., -1:]
        return preds, labels
