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
import collections

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from neural_compressor.common import logger
from neural_compressor.tensorflow.utils.data import default_collate

class ParseDecodeImagenet:
    """Parse features in Example proto.

    Returns:
        tuple of parsed image and label
    """

    def __call__(self, sample):
        """Parse features in example."""
        # Dense features in Example proto.
        feature_map = {
            "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
            "image/class/label": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        }

        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {
                k: sparse_float32
                for k in [
                    "image/object/bbox/xmin",
                    "image/object/bbox/ymin",
                    "image/object/bbox/xmax",
                    "image/object/bbox/ymax",
                ]
            }
        )

        features = tf.io.parse_single_example(serialized=sample, features=feature_map)
        label = tf.cast(features["image/class/label"], dtype=tf.int32)
        image = features["image/encoded"]
        image = tf.image.decode_jpeg(image, channels=3, fancy_upscaling=False, dct_method="INTEGER_FAST")
        return (image, label)


class ResizeCropImagenet(object):
    """Combination of a series of transforms which is applicable to images in Imagenet.

    Args:
        height (int): Height of the result
        width (int): Width of the result
        random_crop (bool, default=False): whether to random crop
        resize_side (int, default=256):desired shape after resize operation
        random_flip_left_right (bool, default=False): whether to random flip left and right
        mean_value (list, default=[0.0,0.0,0.0]):means for each channel
        scale (float, default=1.0):std value

    Returns:
        tuple of processed image and label
    """

    def __init__(
        self,
        height,
        width,
        random_crop=False,
        resize_side=256,
        resize_method="bilinear",
        random_flip_left_right=False,
        mean_value=[0.0, 0.0, 0.0],
        scale=1.0,
        data_format="channels_last",
        subpixels="RGB",
    ):
        """Initialize `TensorflowResizeCropImagenetTransform` class."""
        self.height = height
        self.width = width
        self.mean_value = mean_value
        self.scale = scale
        self.random_crop = random_crop
        self.random_flip_left_right = random_flip_left_right
        self.resize_side = resize_side
        self.resize_method = resize_method
        self.data_format = data_format
        self.subpixels = subpixels

    # sample is (images, labels)
    def __call__(self, sample):
        """Convert `TensorflowResizeCropImagenetTransform` feature."""
        image, label = sample
        shape = tf.shape(input=image)

        height = (
            tf.cast(shape[0], dtype=tf.float32)
            if self.data_format == "channels_last"
            else tf.cast(shape[1], dtype=tf.float32)
        )
        width = (
            tf.cast(shape[1], dtype=tf.float32)
            if self.data_format == "channels_last"
            else tf.cast(shape[2], dtype=tf.float32)
        )
        scale = tf.cond(
            pred=tf.greater(height, width),
            true_fn=lambda: self.resize_side / width,
            false_fn=lambda: self.resize_side / height,
        )

        scale = tf.cast(scale, dtype=tf.float32)
        new_height = tf.cast(tf.math.rint(height * scale), dtype=tf.int32)
        new_width = tf.cast(tf.math.rint(width * scale), dtype=tf.int32)

        if self.subpixels == "BGR" and self.data_format == "channels_first":
            # 'RGB'->'BGR'
            image = tf.cond(
                tf.equal(tf.rank(image), 3),
                lambda: tf.experimental.numpy.moveaxis(image[::-1, ...], 0, -1),
                lambda: tf.experimental.numpy.moveaxis(image[:, ::-1, ...], 1, -1),
            )
        elif self.subpixels == "BGR":
            # 'RGB'->'BGR'
            image = image[..., ::-1]
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [new_height, new_width], method=self.resize_method)
        image = tf.squeeze(image)
        shape = tf.shape(input=image)
        if self.random_crop:
            y0 = tf.random.uniform(shape=[], minval=0, maxval=(shape[0] - self.height + 1), dtype=tf.dtypes.int32)
            x0 = tf.random.uniform(shape=[], minval=0, maxval=(shape[1] - self.width + 1), dtype=tf.dtypes.int32)
        else:
            y0 = (shape[0] - self.height) // 2
            x0 = (shape[1] - self.width) // 2

        image = tf.image.crop_to_bounding_box(image, y0, x0, self.height, self.width)
        image.set_shape([self.height, self.width, 3])
        if self.random_flip_left_right:
            image = tf.image.random_flip_left_right(image)
        means = tf.broadcast_to(self.mean_value, tf.shape(input=image))
        image = (image - means) * self.scale
        return (image, label)


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


class LabelShift(object):
    """Convert label to label - label_shift.

    Args:
        label_shift(int, default=0): number of label shift

    Returns:
        tuple of processed image and label
    """

    def __init__(self, label_shift=0):
        """Initialize `LabelShift` class."""
        self.label_shift = label_shift

    def __call__(self, sample):
        """Convert label to label_shift."""
        images, labels = sample
        if isinstance(labels, np.ndarray):
            labels = labels - self.label_shift
        elif isinstance(labels, list):
            if isinstance(labels[0], tuple):
                labels = [tuple(np.array(label) - self.label_shift) for label in labels]
            elif isinstance(labels[0], np.ndarray):
                labels = [label - self.label_shift for label in labels]
            else:
                labels = np.array(labels) - self.label_shift
                labels = labels.tolist()
        else:
            labels = np.array(labels) - self.label_shift
        return images, labels


class ImageRecordDataset(object):
    """Tensorflow imageNet database in tf record format.

    Please arrange data in this way:
        root/validation-000-of-100
        root/validation-001-of-100
        ...
        root/validation-099-of-100
    The file name needs to follow this pattern: '* - * -of- *'

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    """Configuration for Imagenet dataset."""

    def __new__(cls, root, transform=None, filter=None):
        """Build a new object of TensorflowImageRecord class."""
        from tensorflow.python.platform import gfile  # pylint: disable=no-name-in-module

        glob_pattern = os.path.join(root, "*-*-of-*")
        file_names = gfile.Glob(glob_pattern)
        if not file_names:
            raise ValueError("Found no files in --root matching: {}".format(glob_pattern))

        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave

        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(parallel_interleave(tf.data.TFRecordDataset, cycle_length=len(file_names)))

        if transform is not None:
            transform.transform_list.insert(0, ParseDecodeImagenet())
        else:
            transform = ParseDecodeImagenet()
        ds = ds.map(transform, num_parallel_calls=None)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds


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
    

class TFDataLoader(object):  # pragma: no cover
    """Tensorflow dataloader class.

    In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
    method to do session run, this dataloader is designed to satisfy the usage of feed dict
    in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    Args:
        dataset: obj. wrapper of needed data.
        batch_size: int. batch size
    """

    def __init__(self, dataset, batch_size=1, last_batch="rollover"):
        """Initialize `TFDataDataLoader` class."""
        self.dataset = dataset
        self.last_batch = last_batch
        self.batch_size = batch_size
        dataset = dataset.batch(batch_size)

    def batch(self, batch_size, last_batch="rollover"):
        """Dataset return data per batch."""
        drop_last = False if last_batch == "rollover" else True
        self.batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size, drop_last)

    def __iter__(self):
        """Iterate dataloader."""
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
        )

    def _generate_dataloader(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=None,
        pin_memory=None,
        distributed=False,
    ):
        """Yield data."""
        drop_last = False if last_batch == "rollover" else True

        def check_dynamic_shape(element_spec):
            if isinstance(element_spec, collections.abc.Sequence):
                return any([check_dynamic_shape(ele) for ele in element_spec])
            elif isinstance(element_spec, tf.TensorSpec):
                return True if element_spec.shape.num_elements() is None else False
            else:
                raise ValueError("unrecognized element spec...")

        def squeeze_output(output):
            if isinstance(output, collections.abc.Sequence):
                return [squeeze_output(ele) for ele in output]
            elif isinstance(output, np.ndarray):
                return np.squeeze(output, axis=0)
            else:
                raise ValueError("not supported output format....")

        if tf.executing_eagerly():
            index = 0
            outputs = []
            for iter_tensors in dataset:
                samples = []
                iter_inputs, iter_labels = iter_tensors[0], iter_tensors[1]
                if isinstance(iter_inputs, tf.Tensor):
                    samples.append(iter_inputs.numpy())
                else:
                    samples.append(tuple(iter_input.numpy() for iter_input in iter_inputs))
                if isinstance(iter_labels, tf.Tensor):
                    samples.append(iter_labels.numpy())
                else:
                    samples.append([np.array(l) for l in iter_labels])
                index += 1
                outputs.append(samples)
                if index == batch_size:
                    outputs = default_collate(outputs)
                    yield outputs
                    outputs = []
                    index = 0
            if len(outputs) > 0:
                outputs = default_collate(outputs)
                yield outputs
        else:
            try_single_batch = check_dynamic_shape(dataset.element_spec)
            dataset = dataset.batch(1 if try_single_batch else batch_size, drop_last)
            ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            iter_tensors = ds_iterator.get_next()
            data_config = tf.compat.v1.ConfigProto()
            data_config.use_per_session_threads = 1
            data_config.intra_op_parallelism_threads = 1
            data_config.inter_op_parallelism_threads = 16
            data_sess = tf.compat.v1.Session(config=data_config)
            # pylint: disable=no-name-in-module
            from tensorflow.python.framework.errors_impl import OutOfRangeError

            while True:
                if not try_single_batch:
                    try:
                        outputs = data_sess.run(iter_tensors)
                        yield outputs
                    except OutOfRangeError:
                        data_sess.close()
                        return
                else:
                    try:
                        outputs = []
                        for i in range(0, batch_size):
                            outputs.append(squeeze_output(data_sess.run(iter_tensors)))
                        outputs = default_collate(outputs)
                        yield outputs
                    except OutOfRangeError:
                        if len(outputs) == 0:
                            data_sess.close()
                            return
                        else:
                            outputs = default_collate(outputs)
                            yield outputs
                            data_sess.close()
                            return
