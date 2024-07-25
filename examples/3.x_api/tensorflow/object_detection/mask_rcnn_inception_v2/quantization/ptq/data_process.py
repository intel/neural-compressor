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
import cv2
import collections

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from neural_compressor.common import logger
from neural_compressor.tensorflow.utils.data import default_collate

interpolation_map = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}

category_map = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

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
    

class ResizeWithRatio():
    """Resize image with aspect ratio and pad it to max shape(optional).

    If the image is padded, the label will be processed at the same time.
    The input image should be np.array.

    Args:
        min_dim (int, default=800):
            Resizes the image such that its smaller dimension == min_dim
        max_dim (int, default=1365):
            Ensures that the image longest side doesn't exceed this value
        padding (bool, default=False):
            If true, pads image with zeros so its size is max_dim x max_dim

    Returns:
        tuple of processed image and label
    """

    def __init__(self, min_dim=800, max_dim=1365, padding=False, constant_value=0):
        """Initialize `ResizeWithRatio` class."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.padding = padding
        self.constant_value = constant_value

    def __call__(self, sample):
        """Resize the image with ratio in sample."""
        image, label = sample
        height, width = image.shape[:2]
        scale = 1
        if self.min_dim:
            scale = max(1, self.min_dim / min(height, width))
        if self.max_dim:
            image_max = max(height, width)
            if round(image_max * scale) > self.max_dim:
                scale = self.max_dim / image_max
        if scale != 1:
            image = cv2.resize(image, (round(height * scale), round(width * scale)))

        bbox, str_label, int_label, image_id = label

        if self.padding:
            h, w = image.shape[:2]
            pad_param = [
                [(self.max_dim - h) // 2, self.max_dim - h - (self.max_dim - h) // 2],
                [(self.max_dim - w) // 2, self.max_dim - w - (self.max_dim - w) // 2],
                [0, 0],
            ]
            if not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox)
            resized_box = bbox * [height, width, height, width] * scale
            moved_box = resized_box + [
                (self.max_dim - h) // 2,
                (self.max_dim - w) // 2,
                (self.max_dim - h) // 2,
                (self.max_dim - w) // 2,
            ]
            bbox = moved_box / [self.max_dim, self.max_dim, self.max_dim, self.max_dim]
            image = np.pad(image, pad_param, mode="constant", constant_values=self.constant_value)
        return image, (bbox, str_label, int_label, image_id)


class TensorflowResizeWithRatio():
    """Resize image with aspect ratio and pad it to max shape(optional).

    If the image is padded, the label will be processed at the same time.
    The input image should be np.array or tf.Tensor.

    Args:
        min_dim (int, default=800):
            Resizes the image such that its smaller dimension == min_dim
        max_dim (int, default=1365):
            Ensures that the image longest side doesn't exceed this value
        padding (bool, default=False):
            If true, pads image with zeros so its size is max_dim x max_dim

    Returns:
        tuple of processed image and label
    """

    def __init__(self, min_dim=800, max_dim=1365, padding=False, constant_value=0):
        """Initialize `TensorflowResizeWithRatio` class."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.padding = padding
        self.constant_value = constant_value

    def __call__(self, sample):
        """Resize the image with ratio in sample."""
        image, label = sample
        if isinstance(image, tf.Tensor):
            shape = tf.shape(input=image)
            height = tf.cast(shape[0], dtype=tf.float32)
            width = tf.cast(shape[1], dtype=tf.float32)
            scale = 1
            if self.min_dim:
                scale = tf.maximum(1.0, tf.cast(self.min_dim / tf.math.minimum(height, width), dtype=tf.float32))
            if self.max_dim:
                image_max = tf.cast(tf.maximum(height, width), dtype=tf.float32)
                scale = tf.cond(
                    pred=tf.greater(tf.math.round(image_max * scale), self.max_dim),
                    true_fn=lambda: self.max_dim / image_max,
                    false_fn=lambda: scale,
                )
            image = tf.image.resize(image, (tf.math.round(height * scale), tf.math.round(width * scale)))
            bbox, str_label, int_label, image_id = label

            if self.padding:
                shape = tf.shape(input=image)
                h = tf.cast(shape[0], dtype=tf.float32)
                w = tf.cast(shape[1], dtype=tf.float32)
                pad_param = [
                    [(self.max_dim - h) // 2, self.max_dim - h - (self.max_dim - h) // 2],
                    [(self.max_dim - w) // 2, self.max_dim - w - (self.max_dim - w) // 2],
                    [0, 0],
                ]
                resized_box = bbox * [height, width, height, width] * scale
                moved_box = resized_box + [
                    (self.max_dim - h) // 2,
                    (self.max_dim - w) // 2,
                    (self.max_dim - h) // 2,
                    (self.max_dim - w) // 2,
                ]
                bbox = moved_box / [self.max_dim, self.max_dim, self.max_dim, self.max_dim]
                image = tf.pad(image, pad_param, constant_values=self.constant_value)
        else:
            transform = ResizeWithRatio(self.min_dim, self.max_dim, self.padding)
            image, (bbox, str_label, int_label, image_id) = transform(sample)
        return image, (bbox, str_label, int_label, image_id)


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
        return self._metric

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


class LabelBalanceCOCORecordFilter(object):
    """The label balance filter for COCO Record."""

    def __init__(self, size=1):
        """Initialize the attribute of class."""
        self.size = size

    def __call__(self, image, label):
        """Execute the filter.

        Args:
            image: Not used.
            label: label of a sample.
        """
        return tf.math.equal(len(label[0]), self.size)


class COCOmAPv2(BaseMetric):
    """Compute mean average precision of the detection task."""

    def __init__(
        self,
        anno_path=None,
        iou_thrs="0.5:0.05:0.95",
        map_points=101,
        map_key="DetectionBoxes_Precision/mAP",
        output_index_mapping={"num_detections": -1, "boxes": 0, "scores": 1, "classes": 2},
    ):
        """Initialize the metric.

        Args:
            anno_path: The path of annotation file.
            iou_thrs: Minimal value for intersection over union that allows to make decision
              that prediction bounding box is true positive. You can specify one float value
              between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.
            map_points: The way to calculate mAP. 101 for 101-point interpolated AP, 11 for
              11-point interpolated AP, 0 for area under PR curve.
            map_key: The key that mapping to pycocotools COCOeval.
              Defaults to 'DetectionBoxes_Precision/mAP'.
            output_index_mapping: The output index mapping.
              Defaults to {'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}.
        """
        self.output_index_mapping = output_index_mapping

        if anno_path:
            import os
            import yaml

            assert os.path.exists(anno_path), "Annotation path does not exists!"
            with open(anno_path, "r") as f:
                label_map = yaml.safe_load(f.read())
            self.category_map_reverse = {k: v for k, v in label_map.items()}
        else:
            # label: index
            self.category_map_reverse = {v: k for k, v in category_map.items()}
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1
        self.category_map = category_map
        self.category_id_set = set([cat for cat in self.category_map])  # index
        self.iou_thrs = iou_thrs
        self.map_points = map_points
        self.map_key = map_key

    def update(self, predicts, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            predicts: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight. Defaults to None.
        """
        from coco_tools import ExportSingleImageDetectionBoxesToCoco, ExportSingleImageGroundtruthToCoco

        detections = []
        if "num_detections" in self.output_index_mapping and self.output_index_mapping["num_detections"] > -1:
            for item in zip(*predicts):
                detection = {}
                num = int(item[self.output_index_mapping["num_detections"]])
                detection["boxes"] = np.asarray(item[self.output_index_mapping["boxes"]])[0:num]
                detection["scores"] = np.asarray(item[self.output_index_mapping["scores"]])[0:num]
                detection["classes"] = np.asarray(item[self.output_index_mapping["classes"]])[0:num]
                detections.append(detection)
        else:
            for item in zip(*predicts):
                detection = {}
                detection["boxes"] = np.asarray(item[self.output_index_mapping["boxes"]])
                detection["scores"] = np.asarray(item[self.output_index_mapping["scores"]])
                detection["classes"] = np.asarray(item[self.output_index_mapping["classes"]])
                detections.append(detection)

        bboxes, str_labels, int_labels, image_ids = labels
        labels = []
        if len(int_labels[0]) == 0:
            for str_label in str_labels:
                str_label = [x if type(x) == "str" else x.decode("utf-8") for x in str_label]
                labels.append([self.category_map_reverse[x] for x in str_label])
        elif len(str_labels[0]) == 0:
            for int_label in int_labels:
                labels.append([x for x in int_label])

        for idx, image_id in enumerate(image_ids):
            image_id = image_id if type(image_id) == "str" else image_id.decode("utf-8")
            if image_id in self.image_ids:
                continue
            self.image_ids.append(image_id)

            ground_truth = {}
            ground_truth["boxes"] = np.asarray(bboxes[idx])
            ground_truth["classes"] = np.asarray(labels[idx])

            self.ground_truth_list.extend(
                ExportSingleImageGroundtruthToCoco(
                    image_id=image_id,
                    next_annotation_id=self.annotation_id,
                    category_id_set=self.category_id_set,
                    groundtruth_boxes=ground_truth["boxes"],
                    groundtruth_classes=ground_truth["classes"],
                )
            )
            self.annotation_id += ground_truth["boxes"].shape[0]

            self.detection_list.extend(
                ExportSingleImageDetectionBoxesToCoco(
                    image_id=image_id,
                    category_id_set=self.category_id_set,
                    detection_boxes=detections[idx]["boxes"],
                    detection_scores=detections[idx]["scores"],
                    detection_classes=detections[idx]["classes"],
                )
            )

    def reset(self):
        """Reset the prediction and labels."""
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1

    def result(self):
        """Compute mean average precision.

        Returns:
            The mean average precision score.
        """
        from coco_tools import COCOEvalWrapper, COCOWrapper

        if len(self.ground_truth_list) == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        else:
            groundtruth_dict = {
                "annotations": self.ground_truth_list,
                "images": [{"id": image_id} for image_id in self.image_ids],
                "categories": [{"id": k, "name": v} for k, v in self.category_map.items()],
            }
            coco_wrapped_groundtruth = COCOWrapper(groundtruth_dict)
            coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(self.detection_list)
            box_evaluator = COCOEvalWrapper(
                coco_wrapped_groundtruth,
                coco_wrapped_detections,
                agnostic_mode=False,
                iou_thrs=self.iou_thrs,
                map_points=self.map_points,
            )
            box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
                include_metrics_per_category=False, all_metrics_per_category=False
            )
            box_metrics.update(box_per_category_ap)
            box_metrics = {"DetectionBoxes_" + key: value for key, value in iter(box_metrics.items())}

            return box_metrics[self.map_key]


class ParseDecodeCoco:  # pragma: no cover
    """Helper function for TensorflowModelZooBertDataset.

    Parse the features from sample.
    """

    def __call__(self, sample):
        """Parse the sample data.

        Args:
            sample: Data to be parsed.
        """
        # Dense features in Example proto.
        feature_map = {
            "image/encoded": tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=""),
            "image/object/class/text": tf.compat.v1.VarLenFeature(dtype=tf.string),
            "image/object/class/label": tf.compat.v1.VarLenFeature(dtype=tf.int64),
            "image/source_id": tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=""),
        }
        sparse_float32 = tf.compat.v1.VarLenFeature(dtype=tf.float32)
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

        features = tf.io.parse_single_example(sample, feature_map)

        xmin = tf.expand_dims(features["image/object/bbox/xmin"].values, 0)
        ymin = tf.expand_dims(features["image/object/bbox/ymin"].values, 0)
        xmax = tf.expand_dims(features["image/object/bbox/xmax"].values, 0)
        ymax = tf.expand_dims(features["image/object/bbox/ymax"].values, 0)

        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        encoded_image = features["image/encoded"]
        image_tensor = tf.image.decode_image(encoded_image, channels=3)
        image_tensor.set_shape([None, None, 3])

        str_label = features["image/object/class/text"].values
        int_label = features["image/object/class/label"].values
        image_id = features["image/source_id"]

        return image_tensor, (bbox[0], str_label, int_label, image_id)


class COCORecordDataset(object):
    """Tensorflow COCO dataset in tf record format.

    Root is a full path to tfrecord file, which contains the file name.
    Please use Resize transform when batch_size > 1

    Args: root (str): Root directory of dataset.
          num_cores (int, default=28):The number of input Datasets to interleave from in parallel.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    def __new__(cls, root, num_cores=28, transform=None, filter=filter):
        """Build a new object."""
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(root)
        example = tf.train.SequenceExample()
        for element in record_iterator:
            example.ParseFromString(element)
            break
        feature = example.context.feature
        if (
            len(feature["image/object/class/text"].bytes_list.value) == 0
            and len(feature["image/object/class/label"].int64_list.value) == 0
        ):
            raise ValueError(
                "Tfrecord format is incorrect, please refer\
                'https://github.com/tensorflow/models/blob/master/research/\
                object_detection/dataset_tools/create_coco_tf_record.py' to\
                create correct tfrecord"
            )
        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave

        tfrecord_paths = [root]
        ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)
        ds = ds.apply(
            parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=num_cores,
                block_length=5,
                sloppy=True,
                buffer_output_elements=10000,
                prefetch_input_elements=10000,
            )
        )
        if transform is not None:
            transform.transform_list.insert(0, ParseDecodeCoco())
        else:
            transform = ParseDecodeCoco()
        ds = ds.map(transform, num_parallel_calls=None)
        if filter is not None:
            ds = ds.filter(filter)
        ds = ds.prefetch(buffer_size=1000)
        return ds


class TFDataLoader(object):
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
