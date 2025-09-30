#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import time
import numpy as np
import tensorflow as tf
from absl import app, flags

from neural_compressor.data import DataLoader
from neural_compressor.metric import COCOmAPv2
from neural_compressor.data import LabelBalanceCOCORecordFilter
from neural_compressor.data import ComposeTransform, COCORecordDataset
from neural_compressor.data import TensorflowResizeWithRatio, ParseDecodeCocoTransform

from coco_constants import LABEL_MAP
from utils import non_max_suppression

flags.DEFINE_integer('batch_size', 1, "batch size")

flags.DEFINE_integer('iters', 100, "iterations")

flags.DEFINE_string("ground_truth", None, "ground truth file")

flags.DEFINE_string("input_graph", None, "input graph")

flags.DEFINE_string("output_graph", None, "input graph")

flags.DEFINE_string("config", None, "Neural Compressor config file")

flags.DEFINE_string("dataset_location", None, "Location of Dataset")

flags.DEFINE_float("conf_threshold", 0.5, "confidence threshold")

flags.DEFINE_float("iou_threshold", 0.4, "IoU threshold")

flags.DEFINE_integer("num_intra_threads", 0, "number of intra threads")

flags.DEFINE_integer("num_inter_threads", 1, "number of inter threads")

flags.DEFINE_boolean("tune", False, "whether to run quantization")

flags.DEFINE_boolean("benchmark", False, "whether to run benchmark")

flags.DEFINE_string("mode", 'accuracy', "mode of benchmark")

flags.DEFINE_boolean("profiling", False, "Signal of profiling")

FLAGS = flags.FLAGS


class NMS():
    def __init__(self, conf_threshold, iou_threshold):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, sample):
        preds, labels = sample
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        filtered_boxes = non_max_suppression(preds,
                                             self.conf_threshold,
                                             self.iou_threshold)

        det_boxes = []
        det_scores = []
        det_classes = []
        for cls, bboxs in filtered_boxes.items():
            det_classes.extend([LABEL_MAP[cls + 1]] * len(bboxs))
            for box, score in bboxs:
                rect_pos = box.tolist()
                y_min, x_min = rect_pos[1], rect_pos[0]
                y_max, x_max = rect_pos[3], rect_pos[2]
                height, width = 416, 416
                det_boxes.append(
                    [y_min / height, x_min / width, y_max / height, x_max / width])
                det_scores.append(score)

        if len(det_boxes) == 0:
            det_boxes = np.zeros((0, 4))
            det_scores = np.zeros((0, ))
            det_classes = np.zeros((0, ))

        return [np.array([det_boxes]), np.array([det_scores]), np.array([det_classes])], labels

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model import Model
    model = Model(model)
    model.input_tensor_names = ["inputs"]
    model.output_tensor_names = ["output_boxes:0"]
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    warmup = 5
    iteration = -1
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        iteration = FLAGS.iters
    postprocess = NMS(FLAGS.conf_threshold, FLAGS.iou_threshold)
    metric = COCOmAPv2(map_key='DetectionBoxes_Precision/mAP@.50IOU')

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            inputs = np.array([inputs])
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()

            if FLAGS.mode == 'accuracy':
                predictions, labels = postprocess((predictions, labels))
                metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == iteration:
                break
        latency = np.array(latency_list[warmup:]).mean() / FLAGS.batch_size
        return latency

    eval_dataset = COCORecordDataset(root=FLAGS.dataset_location, filter=LabelBalanceCOCORecordFilter(size=1), \
        transform=ComposeTransform(transform_list=[ParseDecodeCocoTransform(), 
            TensorflowResizeWithRatio(min_dim=416, max_dim=416, padding=True, constant_value=128)]))
    eval_dataloader=DataLoader(framework='tensorflow', dataset=eval_dataset, batch_size=FLAGS.batch_size)

    latency = eval_func(eval_dataloader)
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        print("Batch size = {}".format(FLAGS.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc 

def main(_):
    calib_dataset = COCORecordDataset(root=FLAGS.dataset_location, filter=LabelBalanceCOCORecordFilter(size=1), \
        transform=ComposeTransform(transform_list=[ParseDecodeCocoTransform(), 
                            TensorflowResizeWithRatio(min_dim=416, max_dim=416, padding=True)]))
    calib_dataloader = DataLoader(framework='tensorflow', dataset=calib_dataset, batch_size=FLAGS.batch_size)

    if FLAGS.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig

        config = PostTrainingQuantConfig(
            inputs=["inputs"],
            outputs=["output_boxes"],
            calibration_sampling_size=[2])
        q_model = quantization.fit(model=FLAGS.input_graph, conf=config,
                                   calib_dataloader=calib_dataloader, eval_func=evaluate)
        q_model.save(FLAGS.output_graph)
    elif FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
            fit(FLAGS.input_graph, conf, b_func=evaluate)
        else:
            accuracy = evaluate(FLAGS.input_graph)
            print('Batch size = %d' % FLAGS.batch_size)
            print("Accuracy: %.5f" % accuracy)


if __name__ == '__main__':
    app.run(main)
