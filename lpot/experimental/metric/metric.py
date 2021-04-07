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
from lpot.utils.utility import LazyImport, singleton
from lpot.utils import logger
from sklearn.metrics import accuracy_score
import numpy as np

torch_ignite = LazyImport('ignite')
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')

@singleton
class TensorflowMetrics(object):
    def __init__(self):
        self.metrics = {}
        self.metrics.update(TENSORFLOW_METRICS)

@singleton
class PyTorchMetrics(object):
    def __init__(self):
        self.metrics = {
            "topk": WrapPyTorchMetric(
                torch_ignite.metrics.TopKCategoricalAccuracy),
            "Accuracy": WrapPyTorchMetric(
                torch_ignite.metrics.Accuracy),
            "Loss": WrapPyTorchMetric(
                PyTorchLoss),
        }
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


framework_metrics = {"tensorflow": TensorflowMetrics,
                     "mxnet": MXNetMetrics,
                     "pytorch": PyTorchMetrics,
                     "pytorch_ipex": PyTorchMetrics,
                     "onnxrt_qlinearops": ONNXRTQLMetrics,
                     "onnxrt_integerops": ONNXRTITMetrics}

# user/model specific metrics will be registered here
TENSORFLOW_METRICS = {}
MXNET_METRICS = {}
PYTORCH_METRICS = {}
ONNXRT_QL_METRICS = {}
ONNXRT_IT_METRICS = {}

registry_metrics = {"tensorflow": TENSORFLOW_METRICS,
                    "mxnet": MXNET_METRICS,
                    "pytorch": PYTORCH_METRICS,
                    "pytorch_ipex": PYTORCH_METRICS,
                    "onnxrt_qlinearops": ONNXRT_QL_METRICS,
                    "onnxrt_integerops": ONNXRT_IT_METRICS}


class METRICS(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "pytorch", "pytorch_ipex", "onnxrt_qlinearops",
                             "onnxrt_integerops", "mxnet"), \
                             "framework support tensorflow pytorch mxnet onnxrt"
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
                "mxnet",
                "onnxrt_qlinearops",
                "onnxrt_integerops",
                "pytorch",
                "pytorch_ipex"], "The framework support tensorflow mxnet pytorch onnxrt"

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
    assert all([pred.shape == label.shape for (pred, label) in zip(preds, labels)]), 'shape \
           of labels {} does not match shape of predictions {}'.format(labels.shape, preds.shape)
    return preds, labels

@metric_registry('topk', 'mxnet')
class MxnetTopK(BaseMetric):
    """The class of calculating topk metric, which usually is used in classification.

    Args:
        topk (dict): The dict of topk for configuration.

    """

    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):

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
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("sample num is 0 can't calculate topk")
            return 0
        else:
            return self.num_correct / self.num_sample

@metric_registry('F1', 'tensorflow, pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops')
class F1(BaseMetric):
    def __init__(self):
        self._score_list = []

    def update(self, preds, labels):
        from .f1 import f1_score
        result = f1_score(preds, labels)
        self._score_list.append(result)

    def reset(self):
        self._score_list = []

    def result(self):
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

@metric_registry('Accuracy', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops')
class Accuracy(BaseMetric):
    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def update(self, preds, labels, sample_weight=None):
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
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def result(self):
        correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.sample

class PyTorchLoss():
    def __init__(self):
        self._num_examples = 0
        self._device = torch.device('cpu')
        self._sum = torch.tensor(0.0, device=self._device)

    def reset(self):
        self._num_examples = 0
        self._sum = torch.tensor(0.0, device=self._device)

    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        loss = torch.sum(y_pred)
        self._sum += loss.to(self._device)
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise ValueError("Loss must have at least one example \
                                      before it can be computed.")
        return self._sum.item() / self._num_examples
        
@metric_registry('Loss', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops')
class Loss(BaseMetric):
    def __init__(self):
        self.sample = 0
        self.sum = 0

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _shape_validate(preds, labels)
        self.sample += labels[0].shape[0]
        self.sum += sum([np.sum(pred) for pred in preds])

    def reset(self):
        self.sample = 0
        self.sum = 0

    def result(self):
        return self.sum / self.sample

@metric_registry('MAE', 'tensorflow, pytorch, onnxrt_qlinearops, onnxrt_integerops')
class MAE(BaseMetric):
    def __init__(self, compare_label=True):
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _shape_validate(preds, labels)
        self.label_list.extend(labels)
        self.pred_list.extend(preds)

    def reset(self):
        self.label_list = []
        self.pred_list = []

    def result(self):
        aes = [abs(a-b) for (a,b) in zip(self.label_list, self.pred_list)]
        aes_sum = sum([np.sum(ae) for ae in aes])
        aes_size = sum([ae.size for ae in aes])
        assert aes_size, "predictions shouldn't be none"
        return aes_sum / aes_size

@metric_registry('RMSE', 'tensorflow, pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops')
class RMSE(BaseMetric):
    def __init__(self, compare_label=True):
        self.mse = MSE(compare_label)

    def update(self, preds, labels, sample_weight=None):
        self.mse.update(preds, labels, sample_weight)

    def reset(self):
        self.mse.reset()

    def result(self):
        return np.sqrt(self.mse.result())

@metric_registry('MSE', 'tensorflow, pytorch, onnxrt_qlinearops, onnxrt_integerops')
class MSE(BaseMetric):
    def __init__(self, compare_label=True):
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _shape_validate(preds, labels)
        self.pred_list.extend(preds)
        self.label_list.extend(labels)

    def reset(self):
        self.label_list = []
        self.pred_list = []

    def result(self):
        squares = [(a-b)**2.0 for (a,b) in zip(self.label_list, self.pred_list)]
        squares_sum = sum([np.sum(square) for square in squares])
        squares_size = sum([square.size for square in squares])
        assert squares_size, "predictions should't be None"
        return squares_sum / squares_size

@metric_registry('topk', 'tensorflow')
class TensorflowTopK(BaseMetric):
    """The class of calculating topk metric, which usually is used in classification.

    Args:
        topk (dict): The dict of topk for configuration.

    """

    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):

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
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("sample num is 0 can't calculate topk")
            return 0
        else:
            return self.num_correct / self.num_sample

@metric_registry('topk', 'onnxrt_qlinearops, onnxrt_integerops')
class ONNXRTTopK(BaseMetric):
    """The class of calculating topk metric, which usually is used in classification.

    Args:
        topk (dict): The dict of topk for configuration.

    """

    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):

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
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("sample num is 0 can't calculate topk")
            return 0
        else:
            return self.num_correct / self.num_sample


@metric_registry('COCOmAP', 'tensorflow, onnxrt_qlinearops, onnxrt_integerops')
class TensorflowCOCOMAP(BaseMetric):
    """The class of calculating mAP metric

    """
    def __init__(self, anno_path=None):
        from .coco_label_map import category_map
        if anno_path:
            import os
            import json
            assert os.path.exists(anno_path), 'Annotation path does not exists!'
            label_map = {}
            with open(anno_path) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["name"]] = cnt + 1
            self.category_map_reverse = {k:v for k,v in label_map.items()}
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

    def update(self, predicts, labels, sample_weight=None):
        from .coco_tools import ExportSingleImageGroundtruthToCoco,\
            ExportSingleImageDetectionBoxesToCoco
        bbox, str_label,int_label, image_id = labels
        detection = {}
        if len(predicts) == 3:
            detection['boxes'] = np.asarray(predicts[0][0])
            detection['scores'] = np.asarray(predicts[1][0])
            detection['classes'] = np.asarray(predicts[2][0])
        elif len(predicts) == 4:
            num = int(predicts[0][0])
            detection['boxes'] = np.asarray(predicts[1][0])[0:num]
            detection['scores'] = np.asarray(predicts[2][0])[0:num]
            detection['classes'] = np.asarray(predicts[3][0])[0:num]
        else:
            raise ValueError("Unsupported prediction format!")

        ground_truth = {}
        ground_truth['boxes'] = np.asarray(bbox[0])
        if len(int_label[0]) == 0:
            # convert string label to int index
            str_label = [
                x if type(x) == 'str' else x.decode('utf-8')
                for x in str_label[0]
            ]
            ground_truth['classes'] = np.asarray(
                [self.category_map_reverse[x] for x in str_label])
        elif len(str_label[0]) == 0:
            ground_truth['classes'] = np.asarray(
                [x for x in int_label[0]])
        image_id = image_id[0] if type(
            image_id[0]) == 'str' else image_id[0].decode('utf-8')

        if image_id in self.image_ids:
            return
        self.image_ids.append(image_id)
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
                detection_boxes=detection['boxes'],
                detection_scores=detection['scores'],
                detection_classes=detection['classes']))

    def reset(self):
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1

    def result(self):
        from .coco_tools import COCOWrapper, COCOEvalWrapper
        if len(self.ground_truth_list) == 0:
            logger.warning("sample num is 0 can't calculate mAP")
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
                                                 agnostic_mode=False)
            box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
                include_metrics_per_category=False, all_metrics_per_category=False)
            box_metrics.update(box_per_category_ap)
            box_metrics = {
                'DetectionBoxes_' + key: value
                for key, value in iter(box_metrics.items())
            }

            return box_metrics['DetectionBoxes_Precision/mAP']

@metric_registry('SquadF1', 'tensorflow')
class SquadF1(BaseMetric):
    def __init__(self):
        self._score_list = []

    def update(self, preds, labels, sample_weight=None):
        from .evaluate_squad import evaluate
        result = evaluate(labels, preds)
        self._score_list.append(result['f1'])

    def reset(self):
        self._score_list = []

    def result(self):
        return np.array(self._score_list).mean()
