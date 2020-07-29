#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
#

from __future__ import division
import time
import numpy as np
import tensorflow as tf
import ilit
import logging

from ilit.adaptor.tf_utils.util import write_graph
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

from argparse import ArgumentParser
from coco_detection_evaluator import CocoDetectionEvaluator
from coco_label_map import category_map

IMAGE_SIZE = 300
COCO_NUM_VAL_IMAGES = 4952

tf.compat.v1.disable_eager_execution()

def parse_and_preprocess(serialized_example):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded':
        tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/class/text':
        tf.compat.v1.VarLenFeature(dtype=tf.string),
        'image/source_id':
        tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.compat.v1.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({
        k: sparse_float32
        for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax'
        ]
    })

    features = tf.compat.v1.parse_single_example(serialized_example,
                                                 feature_map)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    encoded_image = features['image/encoded']
    image_tensor = tf.image.decode_image(encoded_image, channels=3)
    image_tensor.set_shape([None, None, 3])

    label = features['image/object/class/text'].values

    image_id = features['image/source_id']

    return image_tensor, bbox[0], label, image_id


class model_infer:
    def __init__(self, args):
        # parse the arguments
        self.args = args

        self.config = tf.compat.v1.ConfigProto()
        self.config.intra_op_parallelism_threads = self.args.num_intra_threads
        self.config.inter_op_parallelism_threads = self.args.num_inter_threads
        self.config.use_per_session_threads = 1

        self.input_layer = 'image_tensor'
        self.output_layers = [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]
        self.load_graph()

        if self.args.batch_size == -1:
            self.args.batch_size = 1

        self.input_tensor = self.infer_graph.get_tensor_by_name(
            self.input_layer + ":0")
        self.output_tensors = [
            self.infer_graph.get_tensor_by_name(x + ":0")
            for x in self.output_layers
        ]

        self.category_map_reverse = {v: k for k, v in category_map.items()}

    def build_data_sess(self):
        data_graph = tf.Graph()
        with data_graph.as_default():
            self.input_images, self.bbox, self.label, self.image_id = self.get_input(
            )
        self.data_sess = tf.compat.v1.Session(graph=data_graph,
                                              config=self.config)

    def load_graph(self):
        print('load graph from: ' + self.args.input_graph)

        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.FastGFile(self.args.input_graph,
                                              'rb') as input_file:
                input_graph_content = input_file.read()
                graph_def.ParseFromString(input_graph_content)
            output_graph = optimize_for_inference(
                graph_def, [self.input_layer], self.output_layers,
                dtypes.uint8.as_datatype_enum, False)
            tf.import_graph_def(output_graph, name='')

    def get_graph(self):
        return self.infer_graph

    def __iter__(self):
        """Enable the generator for q_dataloader

        Yields:
            [Tensor]: images
        """
        data_graph = tf.Graph()
        with data_graph.as_default():
            self.input_images, self.bbox, self.label, self.image_id = self.get_input(
            )

        self.data_sess = tf.compat.v1.Session(graph=data_graph,
                                              config=self.config)
        for i in range(COCO_NUM_VAL_IMAGES):
            input_images = self.data_sess.run([self.input_images])
            yield input_images

    def run_benchmark(self):
        if self.args.data_location:
            print("Inference with real data.")
        else:
            print("Inference with dummy data.")

        with tf.compat.v1.Session(graph=self.infer_graph,
                                  config=self.config) as sess:

            if self.args.data_location:
                self.build_data_sess()
            else:
                input_images = sess.run(
                    tf.random.truncated_normal(
                        [self.args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3],
                        dtype=tf.float32,
                        stddev=10,
                        name='synthetic_images'))

            total_iter = self.args.total_iter
            warmup_iter = self.args.warmup_iter
            ttime = 0.0

            print('total iteration is {0}'.format(str(total_iter)))
            print('warm up iteration is {0}'.format(str(warmup_iter)))

            for step in range(total_iter):
                start_time = time.time()
                if self.args.data_location:
                    input_images = self.data_sess.run([self.input_images])
                    input_images = input_images[0]
                _ = sess.run(self.output_tensors,
                             {self.input_tensor: input_images})
                end_time = time.time()

                duration = end_time - start_time
                if (step + 1) % 10 == 0:
                    print('steps = {0}, {1} sec'.format(
                        str(step), str(duration)))

                if step + 1 > warmup_iter:
                    ttime += duration

            total_batches = total_iter - warmup_iter
            print('Batchsize: {0}'.format(str(self.args.batch_size)))
            print('Time spent per BATCH: {0:10.4f} ms'.format(
                ttime / total_batches * 1000))
            print('Total samples/sec: {0:10.4f} samples/s'.format(
                total_batches * self.args.batch_size / ttime))

    def get_input(self):
        tfrecord_paths = [self.args.data_location]
        ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)

        ds = ds.apply(
            parallel_interleave(tf.data.TFRecordDataset,
                                cycle_length=28,
                                block_length=5,
                                sloppy=True,
                                buffer_output_elements=10000,
                                prefetch_input_elements=10000))
        ds = ds.prefetch(buffer_size=10000)
        ds = ds.apply(
            map_and_batch(map_func=parse_and_preprocess,
                          batch_size=self.args.batch_size,
                          num_parallel_batches=28,
                          num_parallel_calls=None))
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds_iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        images, bbox, label, image_id = ds_iterator.get_next()

        return images, bbox, label, image_id

    def accuracy_check(self, input_graph=None):
        print("Inference for accuracy check.")
        if input_graph:
            self.infer_graph = input_graph
            # Need to reset the input_tensor/output_tensor
            self.input_tensor = self.infer_graph.get_tensor_by_name(
                self.input_layer + ":0")
            self.output_tensors = [
                self.infer_graph.get_tensor_by_name(x + ":0")
                for x in self.output_layers
            ]
        self.build_data_sess()
        evaluator = CocoDetectionEvaluator()
        with tf.compat.v1.Session(graph=self.infer_graph,
                                  config=self.config) as sess:
            iter = 0
            while True:
                print('Run {0} iter'.format(iter))
                iter += 1
                input_images, bbox, label, image_id = self.data_sess.run(
                    [self.input_images, self.bbox, self.label, self.image_id])
                ground_truth = {}
                ground_truth['boxes'] = np.asarray(bbox[0])
                label = [
                    x if type(x) == 'str' else x.decode('utf-8')
                    for x in label[0]
                ]
                ground_truth['classes'] = np.asarray(
                    [self.category_map_reverse[x] for x in label])
                image_id = image_id[0] if type(
                    image_id[0]) == 'str' else image_id[0].decode('utf-8')
                evaluator.add_single_ground_truth_image_info(
                    image_id, ground_truth)
                num, boxes, scores, labels = sess.run(
                    self.output_tensors, {self.input_tensor: input_images})
                detection = {}
                num = int(num[0])
                detection['boxes'] = np.asarray(boxes[0])[0:num]
                detection['scores'] = np.asarray(scores[0])[0:num]
                detection['classes'] = np.asarray(labels[0])[0:num]
                evaluator.add_single_detected_image_info(image_id, detection)
                if iter * self.args.batch_size >= COCO_NUM_VAL_IMAGES:
                    res = evaluator.evaluate()
                    return res['DetectionBoxes_Precision/mAP']

    def run(self):
        if self.args.accuracy_only:
            self.accuracy_check()
        else:
            self.run_benchmark()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                    "parameter is not specified or is -1, the " \
                                    "largest ideal batch size for the model will " \
                                    "be used.",
                            dest="batch_size", type=int, default=-1)

    arg_parser.add_argument('-e',
                            "--inter-op-parallelism-threads",
                            help='The number of inter-thread.',
                            dest='num_inter_threads',
                            type=int,
                            default=0)

    arg_parser.add_argument('-a',
                            "--intra-op-parallelism-threads",
                            help='The number of intra-thread.',
                            dest='num_intra_threads',
                            type=int,
                            default=0)

    arg_parser.add_argument('-g',
                            "--input-graph",
                            help='Specify the input graph.',
                            dest='input_graph')

    arg_parser.add_argument('-d',
                            "--data-location",
                            help='Specify the location of the data. '
                            'If this parameter is not specified, '
                            'the benchmark will use random/dummy data.',
                            dest="data_location",
                            default=None)

    arg_parser.add_argument('-r',
                            "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only',
                            action='store_true')

    arg_parser.add_argument('-i',
                            "--iter",
                            help='For accuracy measurement only.',
                            dest='total_iter',
                            default=1000,
                            type=int)

    arg_parser.add_argument('-w',
                            "--warmup_iter",
                            help='For accuracy measurement only.',
                            dest='warmup_iter',
                            default=200,
                            type=int)
    arg_parser.add_argument('--config', type=str, default='')
    arg_parser.add_argument('--output_model', type=str, default='')
    arg_parser.add_argument('--tune', type=bool, default=True)

    args = arg_parser.parse_args()
    infer = model_infer(args)
    if args.tune:
        at = ilit.Tuner(args.config)
        q_dataloader = ilit.DataLoader('tensorflow', infer)
        output_graph = at.tune(infer.get_graph(),
                            q_dataloader=q_dataloader,
                            eval_func=infer.accuracy_check)
        try:
            write_graph(output_graph.as_graph_def(), args.output_model)
        except Exception as e:
            logging.getLogger().info("Failed to save model due to {}".format(str(e)))
    else:
        infer.run()
