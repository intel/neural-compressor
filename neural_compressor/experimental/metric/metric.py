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

from abc import abstractmethod
from collections import Counter
from neural_compressor.utils.utility import LazyImport, singleton
from neural_compressor.utils import logger
from sklearn.metrics import accuracy_score
import numpy as np
import collections

torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')
transformers = LazyImport('transformers')

@singleton
class TensorflowMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(TENSORFLOW_METRICS)

@singleton
class PyTorchMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(PYTORCH_METRICS)

@singleton
class MXNetMetrics(object):
    def __init__(self):
        self.metrics = {
            "Accuracy": WrapMXNetMetric(mx.metric.Accuracy),
            "MAE": WrapMXNetMetric(mx.metric.MAE),
            "MSE": WrapMXNetMetric(mx.metric.MSE),
            "Loss": WrapMXNetMetric(mx.metric.Loss),
        }
        self.metrics.update(MXNET_METRICS)

@singleton
class ONNXRTQLMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(ONNXRT_QL_METRICS)

@singleton
class ONNXRTITMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(ONNXRT_IT_METRICS)

@singleton
class EngineMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(ENGINE_METRICS)

framework_metrics = {"tensorflow": TensorflowMetrics,
                     "tensorflow_itex": TensorflowMetrics,
                     "mxnet": MXNetMetrics,
                     "pytorch": PyTorchMetrics,
                     "pytorch_ipex": PyTorchMetrics,
                     "pytorch_fx": PyTorchMetrics,
                     "onnxrt_qlinearops": ONNXRTQLMetrics,
                     "onnxrt_integerops": ONNXRTITMetrics,
                     "engine": EngineMetrics}

# user/model specific metrics will be registered here
TENSORFLOW_METRICS = {}
MXNET_METRICS = {}
PYTORCH_METRICS = {}
ONNXRT_QL_METRICS = {}
ONNXRT_IT_METRICS = {}
ENGINE_METRICS = {}

registry_metrics = {"tensorflow": TENSORFLOW_METRICS,
                    "tensorflow_itex": TENSORFLOW_METRICS,
                    "mxnet": MXNET_METRICS,
                    "pytorch": PYTORCH_METRICS,
                    "pytorch_ipex": PYTORCH_METRICS,
                    "pytorch_fx": PYTORCH_METRICS,
                    "onnxrt_qlinearops": ONNXRT_QL_METRICS,
                    "onnxrt_integerops": ONNXRT_IT_METRICS,
                    "engine": ENGINE_METRICS,}


class METRICS(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "tensorflow_itex", "engine",
                            "pytorch", "pytorch_ipex", "pytorch_fx", \
                             "onnxrt_qlinearops", "onnxrt_integerops", "mxnet"), \
                             "framework support tensorflow pytorch mxnet onnxrt engine"
        self.metrics = framework_metrics[framework]().metrics

    def __getitem__(self, metric_type):
        assert metric_type in self.metrics.keys(), "only support metrics in {}".\
            format(self.metrics.keys())

        return self.metrics[metric_type]

    def register(self, name, metric_cls):
        assert name not in self.metrics.keys(), 'registered metric name already exists.'
        self.metrics.update({name: metric_cls})

def metric_registry(metric_type, framework):
    """The class decorator used to register all Metric subclasses.
       cross framework metric is supported by add param as framework='tensorflow, \
                       pytorch, mxnet, onnxrt'

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    def decorator_metric(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in [
                "tensorflow",
                "tensorflow_itex",
                "mxnet",
                "onnxrt_qlinearops",
                "onnxrt_integerops",
                "pytorch",
                "pytorch_ipex",
                "pytorch_fx",
                "engine"], "The framework support tensorflow mxnet pytorch onnxrt engine"

            if metric_type in registry_metrics[single_framework].keys():
                raise ValueError('Cannot have two metrics with the same name')
            registry_metrics[single_framework][metric_type] = cls
        return cls
    return decorator_metric


class BaseMetric(object):
    def __init__(self, metric, single_output=False):
        self._metric_cls = metric
        self._single_output = single_output

    def __call__(self, *args, **kwargs):
        self._metric = self._metric_cls(*args, **kwargs)
        return self

    @abstractmethod
    def update(self, preds, labels=None, sample_weight=None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError

    @property
    def metric(self):
        return self._metric

class WrapPyTorchMetric(BaseMetric):

    def update(self, preds, labels=None, sample_weight=None):
        if self._single_output:
            output = torch.as_tensor(preds)
        else:
            output = (torch.as_tensor(preds), torch.as_tensor(labels))
        self._metric.update(output)

    def reset(self):
        self._metric.reset()

    def result(self):
        return self._metric.compute()


class WrapMXNetMetric(BaseMetric):

    def update(self, preds, labels=None, sample_weight=None):
        preds = mx.nd.array(preds)
        labels = mx.nd.array(labels)
        self._metric.update(labels=labels, preds=preds)

    def reset(self):
        self._metric.reset()

    def result(self):
        acc_name, acc = self._metric.get()
        return acc

class WrapONNXRTMetric(BaseMetric):

    def update(self, preds, labels=None, sample_weight=None):
        preds = np.array(preds)
        labels = np.array(labels)
        self._metric.update(labels=labels, preds=preds)

    def reset(self):
        self._metric.reset()

    def result(self):
        acc_name, acc = self._metric.get()
        return acc

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
    assert label_N == N, 'labels batch size should same with preds'
    labels = labels.reshape([N, -1])
    # one-hot labels will have 2 dimension not equal 1
    if labels.shape[1] != 1:
        labels = labels.argsort()[..., -1:]
    return preds, labels

def _shape_validate(preds, labels):
    assert type(preds) in [int, list, np.ndarray], 'preds must be in int or list, ndarray'
    assert type(labels) in [int, list, np.ndarray], 'labels must be in int or list, ndarray'
    if isinstance(preds, int):
        preds = [np.array([preds])]
    elif isinstance(preds[0], int):
        preds = [np.array(preds)]
    else:
        preds = [np.array(pred) for pred in preds]
    if isinstance(labels, int):
        labels = [np.array([labels])]
    elif isinstance(labels[0], int):
        labels = [np.array(labels)]
    else:
        labels = [np.array(label) for label in labels]
    for (pred, label) in zip(preds, labels):
        assert pred.shape == label.shape, \
            'Shape mismatch, label shape {} vs pred shape {}'.format(label.shape, pred.shape)
    return preds, labels

@metric_registry('F1', 'tensorflow, pytorch, mxnet, onnxrt_qlinearops, \
                 onnxrt_integerops, engine')
class F1(BaseMetric):
    """Computes the F1 score of a binary classification problem.

    """
    def __init__(self):
        self._score_list = []

    def update(self, preds, labels):
        """add preds and labels to storage"""
        from .f1 import f1_score
        result = f1_score(preds, labels)
        self._score_list.append(result)

    def reset(self):
        """clear preds and labels storage"""
        self._score_list = []

    def result(self):
        """calculate metric"""
        return np.array(self._score_list).mean()

def _accuracy_shape_check(preds, labels):
    if isinstance(preds, int):
        preds = [preds]
    preds = np.array(preds)
    if isinstance(labels, int):
        labels = [labels]
    labels = np.array(labels)
    if len(labels.shape) != len(preds.shape) and len(labels.shape)+1 != len(preds.shape):
        raise ValueError(
            'labels must have shape of (batch_size, ..) and preds must have'
            'shape of (batch_size, num_classes, ...) or (batch_size, ..),'
            'but given {} and {}.'.format(labels.shape, preds.shape))
    return preds, labels

def _accuracy_type_check(preds, labels):
   if len(preds.shape) == len(labels.shape)+1:
       num_classes = preds.shape[1]
       if num_classes == 1:
           update_type = 'binary'
       else:
           update_type = 'multiclass'
   elif len(preds.shape) == len(labels.shape):
       if len(preds.shape) == 1 or preds.shape[1] ==1:
           update_type = 'binary'
       else:
           update_type = 'multilabel'
   return update_type

@metric_registry('Accuracy', 'tensorflow, pytorch, onnxrt_qlinearops, \
                onnxrt_integerops, engine')
class Accuracy(BaseMetric):
    """Computes accuracy classification score.

    """
    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _accuracy_shape_check(preds, labels)
        update_type = _accuracy_type_check(preds, labels)
        if update_type == 'binary':
            self.pred_list.extend(preds)
            self.label_list.extend(labels)
            self.sample += labels.shape[0]
        elif update_type == 'multiclass':
            self.pred_list.extend(np.argmax(preds, axis=1).astype('int32'))
            self.label_list.extend(labels)
            self.sample += labels.shape[0]
        elif update_type == 'multilabel':
            #(N, C, ...) -> (N*..., C)
            num_label = preds.shape[1]
            last_dim = len(preds.shape)
            if last_dim-1 != 1:
                trans_list = [0]
                trans_list.extend(list(range(2, len(preds.shape))))
                trans_list.extend([1])
                preds = preds.transpose(trans_list).reshape(-1, num_label)
                labels = labels.transpose(trans_list).reshape(-1, num_label)
            self.sample += preds.shape[0]*preds.shape[1]
            self.pred_list.append(preds)
            self.label_list.append(labels)

    def reset(self):
        """clear preds and labels storage"""
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def result(self):
        """calculate metric"""
        correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.sample

class PyTorchLoss():
    """A dummy metric for directly printing loss, it calculates the average of predictions.

    """
    def __init__(self):
        self._num_examples = 0
        self._device = torch.device('cpu')
        self._sum = torch.tensor(0.0, device=self._device)

    def reset(self):
        """clear preds and labels storage"""
        self._num_examples = 0
        self._sum = torch.tensor(0.0, device=self._device)

    def update(self, output):
        """add preds and labels to storage"""
        y_pred, y = output[0].detach(), output[1].detach()
        loss = torch.sum(y_pred)
        self._sum += loss.to(self._device)
        self._num_examples += y.shape[0]

    def compute(self):
        """calculate metric"""
        if self._num_examples == 0:
            raise ValueError("Loss must have at least one example \
                                      before it can be computed.")
        return self._sum.item() / self._num_examples
        
@metric_registry('Loss', 'tensorflow, pytorch, onnxrt_qlinearops, onnxrt_integerops, engine')
class Loss(BaseMetric):
    """A dummy metric for directly printing loss, it calculates the average of predictions.

    """
    def __init__(self):
        self.sample = 0
        self.sum = 0

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _shape_validate(preds, labels)
        self.sample += labels[0].shape[0]
        self.sum += sum([np.sum(pred) for pred in preds])

    def reset(self):
        """clear preds and labels storage"""
        self.sample = 0
        self.sum = 0

    def result(self):
        """calculate metric"""
        return self.sum / self.sample

@metric_registry('MAE', 'tensorflow, pytorch, onnxrt_qlinearops, onnxrt_integerops, engine')
class MAE(BaseMetric):
    """Computes Mean Absolute Error (MAE) loss.

    Args:
        compare_label (bool): Whether to compare label. False if there are no labels
                              and will use FP32 preds as labels.
    """
    def __init__(self, compare_label=True):
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _shape_validate(preds, labels)
        self.label_list.extend(labels)
        self.pred_list.extend(preds)

    def reset(self):
        """clear preds and labels storage"""
        self.label_list = []
        self.pred_list = []

    def result(self):
        """calculate metric"""
        aes = [abs(a-b) for (a,b) in zip(self.label_list, self.pred_list)]
        aes_sum = sum([np.sum(ae) for ae in aes])
        aes_size = sum([ae.size for ae in aes])
        assert aes_size, "predictions shouldn't be none"
        return aes_sum / aes_size

@metric_registry('RMSE', 'tensorflow, pytorch, mxnet, onnxrt_qlinearops, \
                onnxrt_integerops, engine')
class RMSE(BaseMetric):
    """Computes Root Mean Squared Error (RMSE) loss.

    Args:
        compare_label (bool): Whether to compare label. False if there are no labels
                              and will use FP32 preds as labels.
    """
    def __init__(self, compare_label=True):
        self.mse = MSE(compare_label)

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        self.mse.update(preds, labels, sample_weight)

    def reset(self):
        """clear preds and labels storage"""
        self.mse.reset()

    def result(self):
        """calculate metric"""
        return np.sqrt(self.mse.result())

@metric_registry('MSE', 'tensorflow, pytorch, onnxrt_qlinearops, onnxrt_integerops, engine')
class MSE(BaseMetric):
    """Computes Mean Squared Error (MSE) loss.

    Args:
        compare_label (bool): Whether to compare label. False if there are no labels
                              and will use FP32 preds as labels.
    """
    def __init__(self, compare_label=True):
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _shape_validate(preds, labels)
        self.pred_list.extend(preds)
        self.label_list.extend(labels)

    def reset(self):
        """clear preds and labels storage"""
        self.label_list = []
        self.pred_list = []

    def result(self):
        """calculate metric"""
        squares = [(a-b)**2.0 for (a,b) in zip(self.label_list, self.pred_list)]
        squares_sum = sum([np.sum(square) for square in squares])
        squares_size = sum([square.size for square in squares])
        assert squares_size, "predictions should't be None"
        return squares_sum / squares_size

@metric_registry('topk', 'tensorflow')
class TensorflowTopK(BaseMetric):
    """Computes top k predictions accuracy.

    Args:
        k (int): Number of top elements to look at for computing accuracy.
    """
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _topk_shape_validate(preds, labels)

        labels = labels.reshape([len(labels)])
        with tf.Graph().as_default() as acc_graph:
          topk = tf.nn.in_top_k(predictions=tf.constant(preds, dtype=tf.float32),
                                targets=tf.constant(labels, dtype=tf.int32), k=self.k)
          fp32_topk = tf.cast(topk, tf.float32)
          correct_tensor = tf.reduce_sum(input_tensor=fp32_topk)

          with tf.compat.v1.Session() as acc_sess:
            correct  = acc_sess.run(correct_tensor)

        self.num_sample += len(labels)
        self.num_correct += correct

    def reset(self):
        """clear preds and labels storage"""
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        """calculate metric"""
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        else:
            return self.num_correct / self.num_sample

@metric_registry('topk', 'pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops, engine')
class GeneralTopK(BaseMetric):
    """Computes top k predictions accuracy.

    Args:
        k (int): Number of top elements to look at for computing accuracy.
    """
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        preds, labels = _topk_shape_validate(preds, labels)
        preds = preds.argsort()[..., -self.k:]
        if self.k == 1:
            correct = accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1

        self.num_sample += len(labels)

    def reset(self):
        """clear preds and labels storage"""
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        """calculate metric"""
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        else:
            return self.num_correct / self.num_sample

@metric_registry('mAP', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops, engine')
class TensorflowMAP(BaseMetric):
    """Computes mean average precision.

    Args:
        anno_path (str): Annotation path.
        iou_thrs (float or str): Minimal value for intersection over union that allows to
                                 make decision that prediction bounding box is true positive.
                                 You can specify one float value between 0 to 1 or
                                 string "05:0.05:0.95" for standard COCO thresholds.
        map_points (int): The way to calculate mAP. 101 for 101-point interpolated AP, 11 for
                          11-point interpolated AP, 0 for area under PR curve.
    """
    def __init__(self, anno_path=None, iou_thrs=0.5, map_points=0, \
        map_key='DetectionBoxes_Precision/mAP'):
        from .coco_label_map import category_map
        if anno_path:
            import os
            import yaml
            assert os.path.exists(anno_path), 'Annotation path does not exists!'
            with open(anno_path, 'r') as f:
                label_map = yaml.safe_load(f.read())
            self.category_map_reverse = {k: v for k,v in label_map.items()}
        else:
            # label: index
            self.category_map_reverse = {v: k for k, v in category_map.items()}
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1
        self.category_map = category_map
        self.category_id_set = set(
            [cat for cat in self.category_map]) #index
        self.iou_thrs = iou_thrs
        self.map_points = map_points
        self.map_key = map_key


    def update(self, predicts, labels, sample_weight=None):
        """add preds and labels to storage"""
        from .coco_tools import ExportSingleImageGroundtruthToCoco,\
            ExportSingleImageDetectionBoxesToCoco
        detections = []
        if len(predicts) == 3:
            for bbox, score, cls in zip(*predicts):
                detection = {}
                detection['boxes'] = np.asarray(bbox)
                detection['scores'] = np.asarray(score)
                detection['classes'] = np.asarray(cls)
                detections.append(detection)
        elif len(predicts) == 4:
            for num, bbox, score, cls in zip(*predicts):
                detection = {}
                num = int(num)
                detection['boxes'] = np.asarray(bbox)[0:num]
                detection['scores'] = np.asarray(score)[0:num]
                detection['classes'] = np.asarray(cls)[0:num]
                detections.append(detection)
        else:
            raise ValueError("Unsupported prediction format!")

        bboxes, str_labels,int_labels, image_ids = labels
        labels = []
        if len(int_labels[0]) == 0:
            for str_label in str_labels:
                str_label = [
                    x if type(x) == 'str' else x.decode('utf-8')
                    for x in str_label
                ]
                labels.append([self.category_map_reverse[x] for x in str_label])
        elif len(str_labels[0]) == 0:
            for int_label in int_labels:
                labels.append([x for x in int_label])

        for idx, image_id in enumerate(image_ids):
            image_id = image_id if type(
                image_id) == 'str' else image_id.decode('utf-8')
            if image_id in self.image_ids:
                continue
            self.image_ids.append(image_id)

            ground_truth = {}
            ground_truth['boxes'] = np.asarray(bboxes[idx])
            ground_truth['classes'] = np.asarray(labels[idx])

            self.ground_truth_list.extend(
                ExportSingleImageGroundtruthToCoco(
                    image_id=image_id,
                    next_annotation_id=self.annotation_id,
                    category_id_set=self.category_id_set,
                    groundtruth_boxes=ground_truth['boxes'],
                    groundtruth_classes=ground_truth['classes']))
            self.annotation_id += ground_truth['boxes'].shape[0]

            self.detection_list.extend(
                ExportSingleImageDetectionBoxesToCoco(
                    image_id=image_id,
                    category_id_set=self.category_id_set,
                    detection_boxes=detections[idx]['boxes'],
                    detection_scores=detections[idx]['scores'],
                    detection_classes=detections[idx]['classes']))

    def reset(self):
        """clear preds and labels storage"""
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1

    def result(self):
        """calculate metric"""
        from .coco_tools import COCOWrapper, COCOEvalWrapper
        if len(self.ground_truth_list) == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        else:
            groundtruth_dict = {
                'annotations':
                self.ground_truth_list,
                'images': [{
                    'id': image_id
                } for image_id in self.image_ids],
                'categories': [{
                    'id': k,
                    'name': v
                } for k, v in self.category_map.items()]
            }
            coco_wrapped_groundtruth = COCOWrapper(groundtruth_dict)
            coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
                self.detection_list)
            box_evaluator = COCOEvalWrapper(coco_wrapped_groundtruth,
                                                 coco_wrapped_detections,
                                                 agnostic_mode=False,
                                                 iou_thrs = self.iou_thrs,
                                                 map_points = self.map_points)
            box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
                include_metrics_per_category=False, all_metrics_per_category=False)
            box_metrics.update(box_per_category_ap)
            box_metrics = {
                'DetectionBoxes_' + key: value
                for key, value in iter(box_metrics.items())
            }

            return box_metrics[self.map_key]

@metric_registry('COCOmAP', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops, engine')
class TensorflowCOCOMAP(TensorflowMAP):
    """Computes mean average precision using algorithm in COCO

    Args:
        anno_path (str): Annotation path.
        iou_thrs (float or str): Intersection over union threshold.
                        Set to "0.5:0.05:0.95" for standard COCO thresholds.
        map_points (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP.
    """
    def __init__(self, anno_path=None, iou_thrs=None, map_points=None, \
        map_key='DetectionBoxes_Precision/mAP'):
        super(TensorflowCOCOMAP, self).__init__(anno_path, iou_thrs, map_points, map_key)
        self.iou_thrs = '0.5:0.05:0.95'
        self.map_points = 101

@metric_registry('VOCmAP', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops, engine')
class TensorflowVOCMAP(TensorflowMAP):
    """Computes mean average precision using algorithm in VOC

    Args:
        anno_path (str): Annotation path.
        iou_thrs (float or str): Intersection over union threshold. Set to 0.5.
        map_points (int): The way to calculate mAP. Set to 0 for area under PR curve.
    """
    def __init__(self, anno_path=None, iou_thrs=None, map_points=None, \
        map_key='DetectionBoxes_Precision/mAP'):
        super(TensorflowVOCMAP, self).__init__(anno_path, iou_thrs, map_points, map_key)
        self.iou_thrs = 0.5
        self.map_points = 0

@metric_registry('SquadF1', 'tensorflow, engine')
class SquadF1(BaseMetric):
    """Evaluate for v1.1 of the SQuAD dataset

    """
    def __init__(self):
        self._score_list = []
        # squad metric only work when all data preds collected

    def update(self, preds, labels, sample_weight=None):
        """add preds and labels to storage"""
        if preds:
            from .evaluate_squad import evaluate
            result = evaluate(labels, preds)
            self._score_list.append(result['f1'])

    def reset(self):
        """clear preds and labels storage"""
        self._score_list = []

    def result(self):
        """calculate metric"""
        if len(self._score_list) == 0:
            return 0.
        return np.array(self._score_list).mean()


@metric_registry('mIOU', 'tensorflow, engine')
class mIOU(BaseMetric):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        """add preds and labels to storage"""
        preds = preds.flatten()
        labels = labels.flatten()
        mask = (labels >= 0) & (labels < self.num_classes)
        self.hist += np.bincount(
            self.num_classes * labels[mask].astype(int) +
            preds[mask], minlength=self.num_classes ** 2).reshape(self.num_classes,
            self.num_classes)

    def reset(self):
        """clear preds and labels storage"""
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def result(self):
        """calculate metric"""
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) -
        np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        return mean_iu

@metric_registry('GLUE', 'onnxrt_qlinearops, onnxrt_integerops, engine')
class ONNXRTGLUE(BaseMetric):
    """Computes GLUE score.

    Args:
        task (str, default=mrpc): The name of the task.
                                  Choices include mrpc, qqp, qnli, rte,
                                  sts-b, cola, mnli, wnli.

    """
    def __init__(self, task='mrpc'):
        assert task in ['mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', \
            'mnli', 'wnli'], 'Unsupported task type'
        self.pred_list = None
        self.label_list = None
        self.task = task
        self.return_key = {
            "cola": "mcc",
            "mrpc": "acc",
            "sts-b": "corr",
            "qqp": "acc",
            "mnli": "mnli/acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc"
        }

    def update(self, preds, labels):
        """add preds and labels to storage"""
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        if isinstance(labels, list) and len(labels) == 1:
            labels = labels[0]
        if self.pred_list is None:
            self.pred_list = preds
            self.label_list = labels
        else:
            self.pred_list = np.append(self.pred_list, preds, axis=0)
            self.label_list = np.append(self.label_list, labels, axis=0)

    def reset(self):
        """clear preds and labels storage"""
        self.pred_list = None
        self.label_list = None

    def result(self):
        """calculate metric"""
        output_mode = transformers.glue_output_modes[self.task]

        if output_mode == "classification":
            processed_preds = np.argmax(self.pred_list, axis=1)
        elif output_mode == "regression":
            processed_preds = np.squeeze(self.pred_list)
        result = transformers.glue_compute_metrics(\
            self.task, processed_preds, self.label_list)
        return result[self.return_key[self.task]]
