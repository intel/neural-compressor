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
"""Built-in COCO datasets class for multiple framework backends."""

from PIL import Image

from neural_compressor.utils.utility import LazyImport

from .dataset import Dataset, IterableDataset, dataset_registry

tf = LazyImport("tensorflow")
mx = LazyImport("mxnet")
torch = LazyImport("torch")


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


@dataset_registry(dataset_type="COCORecord", framework="tensorflow, tensorflow_itex", dataset_format="")
class COCORecordDataset(IterableDataset):  # pragma: no cover
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


@dataset_registry(
    dataset_type="COCORaw",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops, pytorch, mxnet, tensorflow, \
                    tensorflow_itex",
    dataset_format="",
)
class COCORaw(Dataset):  # pragma: no cover
    """Coco raw dataset.

    Please arrange data in this way:
        /root/img_dir/1.jpg
        /root/img_dir/2.jpg
        ...
        /root/img_dir/n.jpg
        /root/anno_dir
    Please use Resize transform when batch_size > 1

    Args: root (str): Root directory of dataset.
          img_dir (str, default='val2017'): image file directory.
          anno_dir (str, default='annotations/instances_val2017.json'): annotation file directory.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    def __init__(
        self, root, img_dir="val2017", anno_dir="annotations/instances_val2017.json", transform=None, filter=filter
    ):
        """Initialize the attributes of class."""
        import json
        import os

        import numpy as np
        from pycocotools.coco import COCO

        self.image_list = []
        self.transform = transform
        img_path = os.path.join(root, img_dir)
        anno_path = os.path.join(root, anno_dir)
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        for idx, img_id in enumerate(img_ids):
            img_info = {}
            bboxes = []
            labels = []
            ids = []
            img_detail = coco.loadImgs(img_id)[0]
            ids.append(img_detail["file_name"].encode("utf-8"))
            pic_height = img_detail["height"]
            pic_width = img_detail["width"]

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann["bbox"]
                if len(bbox) == 0:
                    continue
                bbox = [
                    bbox[0] / float(pic_width),
                    bbox[1] / float(pic_height),
                    bbox[2] / float(pic_width),
                    bbox[3] / float(pic_height),
                ]
                bboxes.append([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
                labels.append(coco.cats[ann["category_id"]]["name"].encode("utf8"))
            img_file = os.path.join(img_path, img_detail["file_name"])
            if not os.path.exists(img_file) or len(bboxes) == 0:
                continue

            if filter and not filter(None, bboxes):
                continue

            with Image.open(img_file) as image:
                image = np.array(image.convert("RGB"))
            self.image_list.append(
                (
                    image,
                    [
                        np.array(bboxes),
                        np.array(labels),
                        np.array([]),
                        np.array(img_detail["file_name"].encode("utf-8")),
                    ],
                )
            )

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


@dataset_registry(
    dataset_type="COCONpy",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops, pytorch, mxnet, tensorflow, \
                    tensorflow_itex",
    dataset_format="",
)
class COCONpy(Dataset):  # pragma: no cover
    """COCO npy dataset.

    Please arrange data in this way:
        /root/npy_dir/1.jpg.npy
        /root/npy_dir/2.jpg.npy
        ...
        /root/npy_dir/n.jpg.npy
        /root/anno_dir

    Args: root (str): Root directory of dataset.
          npy_dir (str, default='val2017'): npy file directory.
          anno_dir (str, default='annotations/instances_val2017.json'): annotation file directory.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    def __init__(
        self, root, npy_dir="val2017", anno_dir="annotations/instances_val2017.json", transform=None, filter=None
    ):
        """Initialize the attributes of class."""
        import json
        import os

        import numpy as np
        from pycocotools.coco import COCO

        self.image_list = []
        npy_path = os.path.join(root, npy_dir)
        anno_path = os.path.join(root, anno_dir)
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        for idx, img_id in enumerate(img_ids):
            img_info = {}
            labels = []
            ids = []
            img_detail = coco.loadImgs(img_id)[0]
            ids.append(img_detail["file_name"].encode("utf-8"))
            pic_height = img_detail["height"]
            pic_width = img_detail["width"]

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann["bbox"]
                category_id = ann["category_id"]
                if len(bbox) == 0:
                    continue
                labels.append((np.array(category_id), np.array(bbox)))
            npy_file = os.path.join(npy_path, img_detail["file_name"])
            npy_file = npy_file + ".npy"
            if not os.path.exists(npy_file):
                continue

            image = np.load(npy_file)
            self.image_list.append((image, labels))

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        return sample
