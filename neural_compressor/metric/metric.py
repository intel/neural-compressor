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
"""Neural Compressor metrics."""


from abc import abstractmethod
from ctypes import Union

import numpy as np
from sklearn.metrics import accuracy_score

from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport, singleton

torch = LazyImport("torch")
tf = LazyImport("tensorflow")
mx = LazyImport("mxnet")
transformers = LazyImport("transformers")


class Metric(object):
    """A wrapper of the information needed to construct a Metric.

    The metric class should take the outputs of the model as the metric's inputs,
    neural_compressor built-in metric always take (predictions, labels) as inputs, it's
    recommended to design metric_cls to take (predictions, labels) as inputs.

    Args:
        metric_cls (cls): Should be a instance of sub_class of neural_compressor.metric.BaseMetric
                          or a customer's metric, which takes (predictions, labels) as inputs.
        name (str, optional): Name for metric. Defaults to 'user_metric'.
    """

    def __init__(self, name="user_metric", metric_cls=None, **kwargs):
        """Initialize a Metric with needed information."""
        self.metric_cls = metric_cls
        self.name = name
        self.kwargs = kwargs


@singleton
class TensorflowMetrics(object):
    """Tensorflow metrics collection.

    Attributes:
        metrics: A dict to maintain all metrics for Tensorflow model.
    """

    def __init__(self) -> None:
        """Initialize the metrics collection."""
        self.metrics = {}
        self.metrics.update(TENSORFLOW_METRICS)


@singleton
class PyTorchMetrics(object):
    """PyTorch metrics collection.

    Attributes:
        metrics: A dict to maintain all metrics for PyTorch model.
    """

    def __init__(self) -> None:
        """Initialize the metrics collection."""
        self.metrics = {}
        self.metrics.update(PYTORCH_METRICS)


@singleton
class MXNetMetrics(object):
    """MXNet metrics collection.

    Attributes:
        metrics: A dict to maintain all metrics for MXNet model.
    """

    def __init__(self) -> None:
        """Initialize the metrics collection."""
        from neural_compressor.adaptor.mxnet_utils.util import check_mx_version

        if check_mx_version("2.0.0"):
            import mxnet.gluon.metric as mx_metrics
        else:
            import mxnet.metric as mx_metrics
        self.metrics = {
            "Accuracy": WrapMXNetMetric(mx_metrics.Accuracy),
            "MAE": WrapMXNetMetric(mx_metrics.MAE),
            "MSE": WrapMXNetMetric(mx_metrics.MSE),
            "Loss": WrapMXNetMetric(mx_metrics.Loss),
        }
        self.metrics.update(MXNET_METRICS)


@singleton
class ONNXRTQLMetrics(object):
    """ONNXRT QLinear metrics collection.

    Attributes:
        metrics: A dict to maintain all metrics for ONNXRT QLinear model.
    """

    def __init__(self) -> None:
        """Initialize the metrics collection."""
        self.metrics = {}
        self.metrics.update(ONNXRT_QL_METRICS)


@singleton
class ONNXRTITMetrics(object):
    """ONNXRT Integer metrics collection.

    Attributes:
        metrics: A dict to maintain all metrics for ONNXRT Integer model.
    """

    def __init__(self) -> None:
        """Initialize the metrics collection."""
        self.metrics = {}
        self.metrics.update(ONNXRT_IT_METRICS)


framework_metrics = {
    "tensorflow": TensorflowMetrics,
    "tensorflow_itex": TensorflowMetrics,
    "keras": TensorflowMetrics,
    "mxnet": MXNetMetrics,
    "pytorch": PyTorchMetrics,
    "pytorch_ipex": PyTorchMetrics,
    "pytorch_fx": PyTorchMetrics,
    "onnxrt_qlinearops": ONNXRTQLMetrics,
    "onnxrt_integerops": ONNXRTITMetrics,
    "onnxrt_qdq": ONNXRTQLMetrics,
    "onnxruntime": ONNXRTQLMetrics,
}

# user/model specific metrics will be registered here
TENSORFLOW_METRICS = {}
TENSORFLOW_ITEX_METRICS = {}
KERAS_METRICS = {}
MXNET_METRICS = {}
PYTORCH_METRICS = {}
ONNXRT_QL_METRICS = {}
ONNXRT_IT_METRICS = {}

registry_metrics = {
    "tensorflow": TENSORFLOW_METRICS,
    "tensorflow_itex": TENSORFLOW_ITEX_METRICS,
    "keras": KERAS_METRICS,
    "mxnet": MXNET_METRICS,
    "pytorch": PYTORCH_METRICS,
    "pytorch_ipex": PYTORCH_METRICS,
    "pytorch_fx": PYTORCH_METRICS,
    "onnxrt_qlinearops": ONNXRT_QL_METRICS,
    "onnxrt_qdq": ONNXRT_QL_METRICS,
    "onnxrt_integerops": ONNXRT_IT_METRICS,
    "onnxruntime": ONNXRT_QL_METRICS,
}


class METRICS(object):
    """Intel Neural Compressor Metrics.

    Attributes:
        metrics: The collection of registered metrics for the specified framework.
    """

    def __init__(self, framework: str):
        """Initialize the metrics collection based on the framework name.

        Args:
            framework: The framework name.
        """
        assert framework in (
            "tensorflow",
            "tensorflow_itex",
            "keras",
            "pytorch",
            "pytorch_ipex",
            "pytorch_fx",
            "onnxrt_qdq",
            "onnxrt_qlinearops",
            "onnxrt_integerops",
            "mxnet",
            "onnxruntime",
        ), "framework support tensorflow pytorch mxnet onnxrt"
        self.metrics = framework_metrics[framework]().metrics

    def __getitem__(self, metric_type: str):
        """Get the metric based on the specified type.

        Args:
            metric_type: The metric type.

        Returns:
             The metric with the specified type.
        """
        assert metric_type in self.metrics.keys(), "only support metrics in {}".format(self.metrics.keys())

        return self.metrics[metric_type]

    def register(self, name, metric_cls) -> None:
        """Register a metric.

        Args:
            name: The name of metric.
            metric_cls: The metric class.
        """
        assert name not in self.metrics.keys(), "registered metric name already exists."
        self.metrics.update({name: metric_cls})


def metric_registry(metric_type: str, framework: str):
    """Decorate for registering all Metric subclasses.

    The cross-framework metric is supported by specifying the framework param
    as one of tensorflow, pytorch, mxnet, onnxrt.

    Args:
        metric_type: The metric type.
        framework: The framework name.

    Returns:
        decorator_metric: The function to register metric class.
    """

    def decorator_metric(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(",")]:
            assert single_framework in [
                "tensorflow",
                "tensorflow_itex",
                "keras",
                "mxnet",
                "onnxrt_qlinearops",
                "onnxrt_integerops",
                "onnxrt_qdq",
                "onnxruntime",
                "pytorch",
                "pytorch_ipex",
                "pytorch_fx",
            ], "The framework support tensorflow mxnet pytorch onnxrt"

            if metric_type in registry_metrics[single_framework].keys():
                raise ValueError("Cannot have two metrics with the same name")
            registry_metrics[single_framework][metric_type] = cls
        return cls

    return decorator_metric


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


class WrapPyTorchMetric(BaseMetric):
    """The wrapper of Metric class for PyTorch."""

    def update(self, preds, labels=None, sample_weight=None):
        """Convert the prediction to torch.

        Args:
            preds: The prediction result.
            labels: The reference. Defaults to None.
            sample_weight: The sampling weight. Defaults to None.
        """
        if self._single_output:
            output = torch.as_tensor(preds)
        else:
            output = (torch.as_tensor(preds), torch.as_tensor(labels))
        self._metric.update(output)

    def reset(self):
        """Clear the predictions and labels."""
        self._metric.reset()

    def result(self):
        """Evaluate the difference between predictions and labels."""
        return self._metric.compute()


class WrapMXNetMetric(BaseMetric):
    """The wrapper of Metric class for MXNet."""

    def update(self, preds, labels=None, sample_weight=None):
        """Convert the prediction to MXNet array.

        Args:
            preds: The prediction result.
            labels: The reference. Defaults to None.
            sample_weight: The sampling weight. Defaults to None.
        """
        preds = mx.nd.array(preds)
        labels = mx.nd.array(labels)
        self._metric.update(labels=labels, preds=preds)

    def reset(self):
        """Clear the predictions and labels."""
        self._metric.reset()

    def result(self):
        """Evaluate the difference between predictions and labels.

        Returns:
            acc: The evaluated result.
        """
        acc_name, acc = self._metric.get()
        return acc


class WrapONNXRTMetric(BaseMetric):
    """The wrapper of Metric class for ONNXRT."""

    def update(self, preds, labels=None, sample_weight=None):
        """Convert the prediction to NumPy array.

        Args:
            preds: The prediction result.
            labels: The reference. Defaults to None.
            sample_weight: The sampling weight. Defaults to None.
        """
        preds = np.array(preds)
        labels = np.array(labels)
        self._metric.update(labels=labels, preds=preds)

    def reset(self):
        """Clear the predictions and labels."""
        self._metric.reset()

    def result(self):
        """Evaluate the difference between predictions and labels.

        Returns:
            acc: The evaluated result.
        """
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
    assert label_N == N, "labels batch size should same with preds"
    labels = labels.reshape([N, -1])
    # one-hot labels will have 2 dimension not equal 1
    if labels.shape[1] != 1:
        labels = labels.argsort()[..., -1:]
    return preds, labels


def _shape_validate(preds, labels):
    assert type(preds) in [int, list, np.ndarray], "preds must be in int or list, ndarray"
    assert type(labels) in [int, list, np.ndarray], "labels must be in int or list, ndarray"
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
    for pred, label in zip(preds, labels):
        assert pred.shape == label.shape, "Shape mismatch, label shape {} vs pred shape {}".format(
            label.shape, pred.shape
        )
    return preds, labels


@metric_registry("F1", "tensorflow, tensorflow_itex, pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops")
class F1(BaseMetric):
    """F1 score of a binary classification problem.

    The F1 score is the harmonic mean of the precision and recall.
    It can be computed with the equation:
    F1 = 2 * (precision * recall) / (precision + recall)
    """

    def __init__(self):
        """Initialize the F1 score list."""
        self._score_list = []

    def update(self, preds, labels):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
        """
        from .f1 import f1_score

        if getattr(self, "_hvd", None) is not None:
            gathered_preds_list = self._hvd.allgather_object(preds)
            gathered_labels_list = self._hvd.allgather_object(labels)
            temp_preds_list, temp_labels_list = [], []
            for i in range(0, self._hvd.size()):
                temp_preds_list += gathered_preds_list[i]
                temp_labels_list += gathered_labels_list[i]
            preds = temp_preds_list
            labels = temp_labels_list
        result = f1_score(preds, labels)
        self._score_list.append(result)

    def reset(self):
        """Clear the predictions and labels."""
        self._score_list = []

    def result(self):
        """Compute the F1 score."""
        return np.array(self._score_list).mean()


def _accuracy_shape_check(preds, labels):
    """Check and convert the shape of predictions and labels.

    Args:
        preds: The predictions.
        labels: The labels corresponding to the predictions.

    Returns:
        preds: The predictions in the format of NumPy array.
        labels: The labels in the format of NumPy array.
    """
    if isinstance(preds, int):
        preds = [preds]
    preds = np.array(preds)
    if isinstance(labels, int):
        labels = [labels]
    labels = np.array(labels)
    if len(labels.shape) != len(preds.shape) and len(labels.shape) + 1 != len(preds.shape):
        raise ValueError(
            "labels must have shape of (batch_size, ..) and preds must have"
            "shape of (batch_size, num_classes, ...) or (batch_size, ..),"
            "but given {} and {}.".format(labels.shape, preds.shape)
        )
    return preds, labels


def _accuracy_type_check(preds, labels):
    """Determine the type of prediction.

    Args:
        preds: The predictions.
        labels: The labels corresponding to the predictions.

    Returns:
        update_type: The type of predictions.
    """
    if len(preds.shape) == len(labels.shape) + 1:
        num_classes = preds.shape[1]
        if num_classes == 1:
            update_type = "binary"
        else:
            update_type = "multiclass"
    elif len(preds.shape) == len(labels.shape):
        if len(preds.shape) == 1 or preds.shape[1] == 1:
            update_type = "binary"
        else:
            update_type = "multilabel"
    return update_type


@metric_registry("Accuracy", "tensorflow, tensorflow_itex, pytorch, onnxrt_qlinearops, onnxrt_integerops")
class Accuracy(BaseMetric):
    """The Accuracy for the classification tasks.

    The accuracy score is the proportion of the total number of predictions
    that were correct classified.

    Attributes:
        pred_list: List of prediction to score.
        label_list: List of labels to score.
        sample: The total number of samples.
    """

    def __init__(self):
        """Initialize predictions, labels and sample."""
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        preds, labels = _accuracy_shape_check(preds, labels)
        update_type = _accuracy_type_check(preds, labels)
        if update_type == "binary":
            self.pred_list.extend(preds)
            self.label_list.extend(labels)
            self.sample += labels.shape[0]
        elif update_type == "multiclass":
            self.pred_list.extend(np.argmax(preds, axis=1).astype("int32"))
            self.label_list.extend(labels)
            self.sample += labels.shape[0]
        elif update_type == "multilabel":
            # (N, C, ...) -> (N*..., C)
            num_label = preds.shape[1]
            last_dim = len(preds.shape)
            if last_dim - 1 != 1:
                trans_list = [0]
                trans_list.extend(list(range(2, len(preds.shape))))
                trans_list.extend([1])
                preds = preds.transpose(trans_list).reshape(-1, num_label)
                labels = labels.transpose(trans_list).reshape(-1, num_label)
            self.sample += preds.shape[0] * preds.shape[1]
            self.pred_list.append(preds)
            self.label_list.append(labels)

    def reset(self):
        """Clear the predictions and labels."""
        self.pred_list = []
        self.label_list = []
        self.sample = 0

    def result(self):
        """Compute the accuracy."""
        correct_num = np.sum(np.array(self.pred_list) == np.array(self.label_list))
        if getattr(self, "_hvd", None) is not None:
            allghter_correct_num = sum(self._hvd.allgather_object(correct_num))
            allgather_sample = sum(self._hvd.allgather_object(self.sample))
            return allghter_correct_num / allgather_sample
        return correct_num / self.sample


class PyTorchLoss:
    """A dummy PyTorch Metric.

    A dummy metric that computes the average of predictions and prints it directly.
    """

    def __init__(self):
        """Initialize the number of examples, sum of prediction.

        and device.
        """
        self._num_examples = 0
        self._device = torch.device("cpu")
        self._sum = torch.tensor(0.0, device=self._device)

    def reset(self):
        """Reset the number of samples and total cases to zero."""
        self._num_examples = 0
        self._sum = torch.tensor(0.0, device=self._device)

    def update(self, output):
        """Add the predictions.

        Args:
            output: The predictions.
        """
        y_pred, y = output[0].detach(), output[1].detach()
        loss = torch.sum(y_pred)
        self._sum += loss.to(self._device)
        self._num_examples += y.shape[0]

    def compute(self):
        """Compute the  average of predictions.

        Raises:
            ValueError: There must have at least one example.

        Returns:
            The dummy loss.
        """
        if self._num_examples == 0:
            raise ValueError(
                "Loss must have at least one example \
                                      before it can be computed."
            )
        return self._sum.item() / self._num_examples


@metric_registry("Loss", "tensorflow, tensorflow_itex, pytorch, onnxrt_qlinearops, onnxrt_integerops")
class Loss(BaseMetric):
    """A dummy Metric.

    A dummy metric that computes the average of predictions and prints it directly.

    Attributes:
        sample: The number of samples.
        sum: The sum of prediction.
    """

    def __init__(self):
        """Initialize the number of samples, sum of prediction."""
        self.sample = 0
        self.sum = 0

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        preds, labels = _shape_validate(preds, labels)
        self.sample += labels[0].shape[0]
        self.sum += sum([np.sum(pred) for pred in preds])

    def reset(self):
        """Reset the number of samples and total cases to zero."""
        self.sample = 0
        self.sum = 0

    def result(self):
        """Compute the  average of predictions.

        Returns:
            The dummy loss.
        """
        if getattr(self, "_hvd", None) is not None:
            allgather_sum = sum(self._hvd.allgather_object(self.sum))
            allgather_sample = sum(self._hvd.allgather_object(self.sample))
            return allgather_sum / allgather_sample
        return self.sum / self.sample


@metric_registry("MAE", "tensorflow, tensorflow_itex, pytorch, onnxrt_qlinearops, onnxrt_integerops")
class MAE(BaseMetric):
    """Computes Mean Absolute Error (MAE) loss.

    Mean Absolute Error (MAE) is the mean of the magnitude of
    difference between the predicted and actual numeric values.

    Attributes:
        pred_list: List of prediction to score.
        label_list: List of references corresponding to the prediction result.
        compare_label (bool): Whether to compare label. False if there are no
          labels and will use FP32 preds as labels.
    """

    def __init__(self, compare_label=True):
        """Initialize the list of prediction and labels.

        Args:
            compare_label: Whether to compare label. False if there are no
              labels and will use FP32 preds as labels.
        """
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        preds, labels = _shape_validate(preds, labels)
        self.label_list.extend(labels)
        self.pred_list.extend(preds)

    def reset(self):
        """Clear the predictions and labels."""
        self.label_list = []
        self.pred_list = []

    def result(self):
        """Compute the MAE score.

        Returns:
            The MAE score.
        """
        aes = [abs(a - b) for (a, b) in zip(self.label_list, self.pred_list)]
        aes_sum = sum([np.sum(ae) for ae in aes])
        aes_size = sum([ae.size for ae in aes])
        assert aes_size, "predictions shouldn't be none"
        if getattr(self, "_hvd", None) is not None:
            aes_sum = sum(self._hvd.allgather_object(aes_sum))
            aes_size = sum(self._hvd.allgather_object(aes_size))
        return aes_sum / aes_size


@metric_registry("RMSE", "tensorflow, tensorflow_itex, pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops")
class RMSE(BaseMetric):
    """Computes Root Mean Squared Error (RMSE) loss.

    Attributes:
        mse: The instance of MSE Metric.
    """

    def __init__(self, compare_label=True):
        """Initialize the mse.

        Args:
            compare_label (bool): Whether to compare label. False if there are no labels
              and will use FP32 preds as labels.
        """
        self.mse = MSE(compare_label)

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        self.mse.update(preds, labels, sample_weight)

    def reset(self):
        """Clear the predictions and labels."""
        self.mse.reset()

    def result(self):
        """Compute the RMSE score.

        Returns:
            The RMSE score.
        """
        if getattr(self, "_hvd", None) is not None:
            self.mse._hvd = self._hvd
        return np.sqrt(self.mse.result())


@metric_registry("MSE", "tensorflow, tensorflow_itex, pytorch, onnxrt_qlinearops, onnxrt_integerops")
class MSE(BaseMetric):
    """Computes Mean Squared Error (MSE) loss.

    Mean Squared Error(MSE) represents the average of the squares of errors.
    For example, the average squared difference between the estimated values
    and the actual values.

    Attributes:
        pred_list: List of prediction to score.
        label_list: List of references corresponding to the prediction result.
        compare_label (bool): Whether to compare label. False if there are no labels
                              and will use FP32 preds as labels.
    """

    def __init__(self, compare_label=True):
        """Initialize the list of prediction and labels.

        Args:
            compare_label: Whether to compare label. False if there are no
              labels and will use FP32 preds as labels.
        """
        self.label_list = []
        self.pred_list = []
        self.compare_label = compare_label

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        preds, labels = _shape_validate(preds, labels)
        self.pred_list.extend(preds)
        self.label_list.extend(labels)

    def reset(self):
        """Clear the predictions and labels."""
        self.label_list = []
        self.pred_list = []

    def result(self):
        """Compute the MSE score.

        Returns:
            The MSE score.
        """
        squares = [(a - b) ** 2.0 for (a, b) in zip(self.label_list, self.pred_list)]
        squares_sum = sum([np.sum(square) for square in squares])
        squares_size = sum([square.size for square in squares])
        assert squares_size, "predictions shouldn't be None"
        if getattr(self, "_hvd", None) is not None:
            squares_sum = sum(self._hvd.allgather_object(squares_sum))
            squares_size = sum(self._hvd.allgather_object(squares_size))
        return squares_sum / squares_size


@metric_registry("topk", "tensorflow, tensorflow_itex")
class TensorflowTopK(BaseMetric):
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
        # extract the contents from tf.Tensor
        if not isinstance(labels, int) and len(labels) > 0 and isinstance(labels[0], tf.Tensor):
            temp_labels = []
            for label_tensor in labels:
                label_contents = label_tensor.numpy()
                temp_labels.append(label_contents)
            labels = temp_labels

        preds, labels = _topk_shape_validate(preds, labels)

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
        elif getattr(self, "_hvd", None) is not None:
            allgather_num_correct = sum(self._hvd.allgather_object(self.num_correct))
            allgather_num_sample = sum(self._hvd.allgather_object(self.num_sample))
            return allgather_num_correct / allgather_num_sample
        return self.num_correct / self.num_sample


@metric_registry("topk", "pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops")
class GeneralTopK(BaseMetric):
    """Compute Top-k Accuracy classification score.

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
        preds, labels = _topk_shape_validate(preds, labels)
        preds = preds.argsort()[..., -self.k :]
        if self.k == 1:
            correct = accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype("int32")
                if l in p:
                    self.num_correct += 1

        self.num_sample += len(labels)

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
        elif getattr(self, "_hvd", None) is not None:
            allgather_num_correct = sum(self._hvd.allgather_object(self.num_correct))
            allgather_num_sample = sum(self._hvd.allgather_object(self.num_sample))
            return allgather_num_correct / allgather_num_sample
        return self.num_correct / self.num_sample


@metric_registry("COCOmAPv2", "tensorflow, tensorflow_itex, onnxrt_qlinearops, onnxrt_integerops")
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
        from .coco_label_map import category_map

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
        from .coco_tools import ExportSingleImageDetectionBoxesToCoco, ExportSingleImageGroundtruthToCoco

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
        from .coco_tools import COCOEvalWrapper, COCOWrapper

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


@metric_registry("mAP", "tensorflow, tensorflow_itex, onnxrt_qlinearops, onnxrt_integerops")
class TensorflowMAP(BaseMetric):
    """Computes mean average precision."""

    def __init__(self, anno_path=None, iou_thrs=0.5, map_points=0, map_key="DetectionBoxes_Precision/mAP"):
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
        """
        from .coco_label_map import category_map

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
            sample_weight: The sample weight.
        """
        if getattr(self, "_hvd", None) is not None:
            raise NotImplementedError("Metric TensorflowMAP currently do not support distributed inference.")

        from .coco_tools import ExportSingleImageDetectionBoxesToCoco, ExportSingleImageGroundtruthToCoco

        detections = []
        if len(predicts) == 3:
            for bbox, score, cls in zip(*predicts):
                detection = {}
                detection["boxes"] = np.asarray(bbox)
                detection["scores"] = np.asarray(score)
                detection["classes"] = np.asarray(cls)
                detections.append(detection)
        elif len(predicts) == 4:
            for num, bbox, score, cls in zip(*predicts):
                detection = {}
                num = int(num)
                detection["boxes"] = np.asarray(bbox)[0:num]
                detection["scores"] = np.asarray(score)[0:num]
                detection["classes"] = np.asarray(cls)[0:num]
                detections.append(detection)
        else:
            raise ValueError("Unsupported prediction format!")

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
        from .coco_tools import COCOEvalWrapper, COCOWrapper

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


@metric_registry("COCOmAP", "tensorflow, tensorflow_itex, onnxrt_qlinearops, onnxrt_integerops")
class TensorflowCOCOMAP(TensorflowMAP):
    """Computes mean average precision using algorithm in COCO."""

    def __init__(self, anno_path=None, iou_thrs=None, map_points=None, map_key="DetectionBoxes_Precision/mAP"):
        """Initialize the iou threshold and max points.

        Args:
            anno_path: The path of annotation file.
            iou_thrs: Minimal value for intersection over union that allows to make decision
              that prediction bounding box is true positive. You can specify one float value
              between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.
            map_points: The way to calculate mAP. 101 for 101-point interpolated AP, 11 for
              11-point interpolated AP, 0 for area under PR curve.
            map_key: The key that mapping to pycocotools COCOeval.
              Defaults to 'DetectionBoxes_Precision/mAP'.
        """
        super(TensorflowCOCOMAP, self).__init__(anno_path, iou_thrs, map_points, map_key)
        self.iou_thrs = "0.5:0.05:0.95"
        self.map_points = 101


@metric_registry("VOCmAP", "tensorflow, tensorflow_itex, onnxrt_qlinearops, onnxrt_integerops")
class TensorflowVOCMAP(TensorflowMAP):
    """Computes mean average precision using algorithm in VOC."""

    def __init__(self, anno_path=None, iou_thrs=None, map_points=None, map_key="DetectionBoxes_Precision/mAP"):
        """Initialize the iou threshold and max points.

        Args:
            anno_path: The path of annotation file.
            iou_thrs: Minimal value for intersection over union that allows to make decision
              that prediction bounding box is true positive. You can specify one float value
              between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.
            map_points: The way to calculate mAP. 101 for 101-point interpolated AP, 11 for
              11-point interpolated AP, 0 for area under PR curve.
            map_key: The key that mapping to pycocotools COCOeval.
              Defaults to 'DetectionBoxes_Precision/mAP'.
        """
        super(TensorflowVOCMAP, self).__init__(anno_path, iou_thrs, map_points, map_key)
        self.iou_thrs = 0.5
        self.map_points = 0


@metric_registry("SquadF1", "tensorflow, tensorflow_itex")
class SquadF1(BaseMetric):
    """Evaluate for v1.1 of the SQuAD dataset."""

    def __init__(self):
        """Initialize the score list."""
        self._score_list = []  # squad metric only work when all data preds collected

    def update(self, preds, labels, sample_weight=None):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight.
        """
        if preds:
            from .evaluate_squad import evaluate

            if getattr(self, "_hvd", None) is not None:
                gathered_preds_list = self._hvd.allgather_object(preds)
                gathered_labels_list = self._hvd.allgather_object(labels)
                temp_preds_list, temp_labels_list = [], []
                for i in range(0, self._hvd.size()):
                    temp_preds_list += gathered_preds_list[i]
                    temp_labels_list += gathered_labels_list[i]
                preds = temp_preds_list
                labels = temp_labels_list
            result = evaluate(labels, preds)
            self._score_list.append(result["f1"])

    def reset(self):
        """Reset the score list."""
        self._score_list = []

    def result(self):
        """Compute F1 score."""
        if len(self._score_list) == 0:
            return 0.0
        return np.array(self._score_list).mean()


@metric_registry("mIOU", "tensorflow, tensorflow_itex")
class mIOU(BaseMetric):
    """Compute the mean IOU(Intersection over Union) score."""

    def __init__(self, num_classes=21):
        """Initialize the number of classes.

        Args:
            num_classes: The number of classes.
        """
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
        """
        preds = preds.flatten()
        labels = labels.flatten()
        p_dtype = preds.dtype
        l_dtype = labels.dtype
        if getattr(self, "_hvd", None) is not None:
            preds = self._hvd.allgather_object(preds)
            labels = self._hvd.allgather_object(labels)
            preds_list, labels_list = np.array([], dtype=p_dtype), np.array([], dtype=l_dtype)
            for i in range(self._hvd.size()):
                preds_list = np.append(preds_list, preds[i])
                labels_list = np.append(labels_list, labels[i])
            preds, labels = preds_list, labels_list
        mask = (labels >= 0) & (labels < self.num_classes)
        self.hist += np.bincount(
            self.num_classes * labels[mask].astype(int) + preds[mask], minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def reset(self):
        """Reset the hist."""
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def result(self):
        """Compute mean IOU.

        Returns:
            The mean IOU score.
        """
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        return mean_iu


@metric_registry("GLUE", "onnxrt_qlinearops, onnxrt_integerops")
class ONNXRTGLUE(BaseMetric):
    """Compute the GLUE score."""

    def __init__(self, task="mrpc"):
        """Initialize the metric.

        Args:
            task:The name of the task (Choices: mrpc, qqp, qnli, rte,
              sts-b, cola, mnli, wnli.).
        """
        assert task in ["mrpc", "qqp", "qnli", "rte", "sts-b", "cola", "mnli", "wnli", "sst-2"], "Unsupported task type"
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
            "wnli": "acc",
            "sst-2": "acc",
        }

    def update(self, preds, labels):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
        """
        if getattr(self, "_hvd", None) is not None:
            raise NotImplementedError("Metric ONNXRTGLUE currently do not support distributed inference.")
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
        """Reset the prediction and labels."""
        self.pred_list = None
        self.label_list = None

    def result(self):
        """Compute the GLUE score."""
        output_mode = transformers.glue_output_modes[self.task]

        if output_mode == "classification":
            processed_preds = np.argmax(self.pred_list, axis=1)
        elif output_mode == "regression":
            processed_preds = np.squeeze(self.pred_list)
        result = transformers.glue_compute_metrics(self.task, processed_preds, self.label_list)
        return result[self.return_key[self.task]]


@metric_registry("ROC", "pytorch")
class ROC(BaseMetric):
    """Computes ROC score."""

    def __init__(self, task="dlrm"):
        """Initialize the metric.

        Args:
            task:The name of the task (Choices: dlrm, dien, wide_deep.).
        """
        assert task in ["dlrm", "dien", "wide_deep"], "Unsupported task type"
        self.pred_list = None
        self.label_list = None
        self.task = task
        self.return_key = {
            "dlrm": "acc",
            "dien": "acc",
            "wide_deep": "acc",
        }

    def update(self, preds, labels):
        """Add the predictions and labels.

        Args:
            preds: The predictions.
            labels: The labels corresponding to the predictions.
        """
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
        """Reset the prediction and labels."""
        self.pred_list = None
        self.label_list = None

    def result(self):
        """Compute the ROC score."""
        import sklearn.metrics

        scores = np.squeeze(self.pred_list)
        targets = np.squeeze(self.label_list)
        roc_auc = sklearn.metrics.roc_auc_score(targets, scores)
        acc = sklearn.metrics.accuracy_score(targets, np.round(scores))
        return acc


def register_customer_metric(user_metric, framework):
    """Register customer metric class or a dict of built-in metric configures.

    1. neural_compressor have many built-in metrics,
       user can pass a metric configure dict to tell neural compressor what metric will be use.
       You also can set multi-metrics to evaluate the performance of a specific model.
            Single metric:
                {topk: 1}
            Multi-metrics:
                {topk: 1,
                 MSE: {compare_label: False},
                 weight: [0.5, 0.5],
                 higher_is_better: [True, False]
                }
    For the built-in metrics, please refer to below link:
    https://github.com/intel/neural-compressor/blob/master/docs/source/metric.md#supported-built-in-metric-matrix.

    2. User also can get the built-in metrics by neural_compressor.Metric:
        Metric(name="topk", k=1)
    3. User also can set specific metric through this api. The metric class should take the outputs of the model or
       postprocess(if have) as inputs, neural_compressor built-in metric always take(predictions, labels)
       as inputs for update, and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric.

    Args:
        user_metric(neural_compressor.metric.Metric or a dict of built-in metric configurations):
            The object of Metric or a dict of built-in metric configurations.

        framework: framework, such as: tensorflow, pytorch......
    """
    if isinstance(user_metric, dict):
        metric_cfg = user_metric
    else:
        if isinstance(user_metric, Metric):
            if user_metric.metric_cls is None:
                name = user_metric.name
                metric_cls = METRICS(framework).metrics[name]
                metric_cfg = {name: {**user_metric.kwargs}}
                return metric_cfg
            else:
                name = user_metric.name
                metric_cls = user_metric.metric_cls
                metric_cfg = {name: {**user_metric.kwargs}}
        else:
            for i in ["reset", "update", "result"]:
                assert hasattr(user_metric, i), "Please realise {} function" "in user defined metric".format(i)
            metric_cls = type(user_metric).__name__
            name = "user_" + metric_cls
            metric_cfg = {name: id(user_metric)}
        metrics = METRICS(framework)
        metrics.register(name, metric_cls)
    return metric_cfg
