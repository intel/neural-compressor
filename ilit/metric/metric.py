from abc import abstractmethod
from ilit.utils.utility import LazyImport, singleton
from ..utils import logger
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

torch_ignite = LazyImport('ignite')
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')

@singleton
class TensorflowMetrics(object):
    def __init__(self):
        self.metrics = {
            "Accuracy":WrapTensorflowMetric(tf.keras.metrics.Accuracy),
            "Sum":WrapTensorflowMetric(tf.keras.metrics.Sum, True),
            "Mean":WrapTensorflowMetric(tf.keras.metrics.Mean, True),
            "MeanRelativeError":WrapTensorflowMetric(tf.keras.metrics.MeanRelativeError),
            "BinaryAccuracy":WrapTensorflowMetric(tf.keras.metrics.BinaryAccuracy),
            "CategoricalAccuracy":WrapTensorflowMetric(tf.keras.metrics.CategoricalAccuracy),
            "SparseCategoricalAccuracy":WrapTensorflowMetric(tf.keras.metrics.SparseCategoricalAccuracy),
            "TopKCategoricalAccuracy":WrapTensorflowMetric(tf.keras.metrics.TopKCategoricalAccuracy),
            "SparseTopKCategoricalAccuracy":WrapTensorflowMetric(tf.keras.metrics.SparseTopKCategoricalAccuracy),
            "FalsePositives":WrapTensorflowMetric(tf.keras.metrics.FalsePositives),
            "FalseNegatives":WrapTensorflowMetric(tf.keras.metrics.FalseNegatives),
            "TrueNegatives":WrapTensorflowMetric(tf.keras.metrics.TrueNegatives),
            "TruePositives":WrapTensorflowMetric(tf.keras.metrics.TruePositives),
            "Precision":WrapTensorflowMetric(tf.keras.metrics.Precision),
            "Recall":WrapTensorflowMetric(tf.keras.metrics.Recall),
            "SensitivityAtSpecificity":WrapTensorflowMetric(tf.keras.metrics.SensitivityAtSpecificity),
            "SpecificityAtSensitivity":WrapTensorflowMetric(tf.keras.metrics.SpecificityAtSensitivity),
            "AUC":WrapTensorflowMetric(tf.keras.metrics.AUC),
            "CosineSimilarity":WrapTensorflowMetric(tf.keras.metrics.CosineSimilarity),
            "MeanAbsoluteError":WrapTensorflowMetric(tf.keras.metrics.MeanAbsoluteError),
            "MeanAbsolutePercentageError":WrapTensorflowMetric(tf.keras.metrics.MeanAbsolutePercentageError),
            "MeanSquaredError":WrapTensorflowMetric(tf.keras.metrics.MeanSquaredError),
            "MeanSquaredLogarithmicError":WrapTensorflowMetric(tf.keras.metrics.MeanSquaredLogarithmicError),
            "Hinge":WrapTensorflowMetric(tf.keras.metrics.Hinge),
            "SquaredHinge":WrapTensorflowMetric(tf.keras.metrics.SquaredHinge),
            "CategoricalHinge":WrapTensorflowMetric(tf.keras.metrics.CategoricalHinge),
            "RootMeanSquaredError":WrapTensorflowMetric(tf.keras.metrics.RootMeanSquaredError),
            "LogCoshError":WrapTensorflowMetric(tf.keras.metrics.LogCoshError),
            "Poisson":WrapTensorflowMetric(tf.keras.metrics.Poisson),
            "KLDivergence":WrapTensorflowMetric(tf.keras.metrics.KLDivergence),
            "SparseCategoricalCrossentropy":WrapTensorflowMetric(tf.keras.metrics.SparseCategoricalCrossentropy),
            "CategoricalCrossentropy":WrapTensorflowMetric(tf.keras.metrics.CategoricalCrossentropy),
            "BinaryCrossentropy":WrapTensorflowMetric(tf.keras.metrics.BinaryCrossentropy),
            "MeanTensor":WrapTensorflowMetric(tf.keras.metrics.MeanTensor, True),
        }
        self.metrics.update(TENSORFLOWMETRICS)

@singleton
class PyTorchMetrics(object):
    def __init__(self):
        self.metrics = {
            "Accuracy":WrapPyTorchMetric(torch_ignite.metrics.Accuracy),
            "Loss":WrapPyTorchMetric(torch_ignite.metrics.Loss),
            "MeanAbsoluteError":WrapPyTorchMetric(torch_ignite.metrics.MeanAbsoluteError),
            "MeanPairwiseDistance":WrapPyTorchMetric(torch_ignite.metrics.MeanPairwiseDistance),
            "MeanSquaredError":WrapPyTorchMetric(torch_ignite.metrics.MeanSquaredError),
            # "TopKCategoricalAccuracy":WrapPyTorchMetric(torch_ignite.metrics.TopKCategoricalAccuracy),
            "topk":WrapPyTorchMetric(torch_ignite.metrics.TopKCategoricalAccuracy),
            "Average":WrapPyTorchMetric(torch_ignite.metrics.Average, True),
            "GeometricAverage":WrapPyTorchMetric(torch_ignite.metrics.GeometricAverage, True),
            "ConfusionMatrix":WrapPyTorchMetric(torch_ignite.metrics.ConfusionMatrix),
            # IoU and mIoU are funtions, while call the function it return a MetricsLambda class 
            "IoU":WrapPyTorchMetric(torch_ignite.metrics.IoU),
            "mIoU":WrapPyTorchMetric(torch_ignite.metrics.mIoU),
            "DiceCoefficient":WrapPyTorchMetric(torch_ignite.metrics.DiceCoefficient),

            "MetricsLambda":WrapPyTorchMetric(torch_ignite.metrics.MetricsLambda),
            "EpochMetric":WrapPyTorchMetric(torch_ignite.metrics.EpochMetric),
            "Fbeta":WrapPyTorchMetric(torch_ignite.metrics.Fbeta),
            "Precision":WrapPyTorchMetric(torch_ignite.metrics.Precision),
            "Recall":WrapPyTorchMetric(torch_ignite.metrics.Recall),
            "RootMeanSquaredError":WrapPyTorchMetric(torch_ignite.metrics.RootMeanSquaredError),
            "RunningAverage":WrapPyTorchMetric(torch_ignite.metrics.RunningAverage),
            "VariableAccumulation":WrapPyTorchMetric(torch_ignite.metrics.VariableAccumulation),
            "Frequency":WrapPyTorchMetric(torch_ignite.metrics.Frequency, True),
        }
        self.metrics.update(PYTORCHMETRICS)

@singleton
class MXNetMetrics(object):
    def __init__(self):
        self.metrics = {
            "Accuracy":WrapMXNetMetric(mx.metric.Accuracy),
            "TopKAccuracy":WrapMXNetMetric(mx.metric.TopKAccuracy),
            "F1":WrapMXNetMetric(mx.metric.F1),
            # "Fbeta":WrapMXNetMetric(mx.metric.Fbeta),
            # "BinaryAccuracy":WrapMXNetMetric(mx.metric.BinaryAccuracy),
            "MCC":WrapMXNetMetric(mx.metric.MCC),
            "MAE":WrapMXNetMetric(mx.metric.MAE),
            "MSE":WrapMXNetMetric(mx.metric.MSE),
            "RMSE":WrapMXNetMetric(mx.metric.RMSE),
            # "MeanPairwiseDistance":WrapMXNetMetric(mx.metric.MeanPairwiseDistance),
            # "MeanCosineSimilarity":WrapMXNetMetric(mx.metric.MeanCosineSimilarity),
            "CrossEntropy":WrapMXNetMetric(mx.metric.CrossEntropy),
            "Perplexity":WrapMXNetMetric(mx.metric.Perplexity),
            "NegativeLogLikelihood":WrapMXNetMetric(mx.metric.NegativeLogLikelihood),
            "PearsonCorrelation":WrapMXNetMetric(mx.metric.PearsonCorrelation),
            "PCC":WrapMXNetMetric(mx.metric.PCC),
            "Loss":WrapMXNetMetric(mx.metric.Loss),
        }
        self.metrics.update(MXNETMETRICS)


framework_metrics = {"tensorflow":TensorflowMetrics, 
                     "mxnet":MXNetMetrics,
                     "pytorch":PyTorchMetrics,}

class METRICS(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "pytorch", "mxnet"), "framework support tensorflow pytorch mxnet"
        self.metrics = framework_metrics[framework]().metrics 

    def __getitem__(self, metric_type):
        assert metric_type in self.metrics.keys(), "only support metrics in {}".format(self.metrics.keys())

        return self.metrics[metric_type] 

# user/model specific metrics will be registered here
TENSORFLOWMETRICS = {}
MXNETMETRICS = {}
PYTORCHMETRICS = {}

registry_metrics = {"tensorflow":TENSORFLOWMETRICS,
                    "mxnet":MXNETMETRICS,
                    "pytorch":PYTORCHMETRICS,}

def metric_registry(metric_type, framework):
    """The class decorator used to register all Metric subclasses.
       cross framework metric is supported by add param as framework='tensorflow, pytorch, mxnet'

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """    
    def decorator_metric(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in ["tensorflow", "mxnet", "pytorch"], "The framework support tensorflow mxnet pytorch"
            
            if metric_type in registry_metrics[single_framework].keys():
                raise ValueError('Cannot have two metrics with the same name')
            registry_metrics[single_framework][metric_type] = cls
        return cls
    return decorator_metric


class Metric(object):
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

class WrapTensorflowMetric(Metric):

    def update(self, preds, labels=None, sample_weight=None):
        if self._single_output:
            _ = self._metric.update_state(values=preds,
                                          sample_weight=sample_weight)
        else:
            _ = self._metric.update_state(y_true=labels,
                                          y_pred=preds,
                                          sample_weight=sample_weight)

    def reset(self):
        self._metric.reset_states() 

    def result(self):
        return self._metric.result()

class WrapPyTorchMetric(Metric):

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

class WrapMXNetMetric(Metric):

    def update(self, preds, labels=None, sample_weight=None):
        preds = mx.nd.array(preds)
        labels = mx.nd.array(labels)
        self._metric.update(labels=labels, preds=preds)

    def reset(self):
        self._metric.reset() 

    def result(self):
        acc_name, acc = self._metric.get()
        return acc

@metric_registry('topk', 'tensorflow, mxnet')
class TopK(Metric):
    """The class of calculating topk metric, which usually is used in classification.

    Args:
        topk (dict): The dict of topk for configuration.

    """
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):

        preds = preds.argsort()[..., -self.k:]
        preds = np.squeeze(preds)
        if self.k == 1:
            correct = accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct
            self.num_sample += labels.shape[0]

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1
            self.num_sample += labels.shape[0]

    def reset(self):
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("sample num is 0 can't calculate topk")
            return 0
        else:
            return self.num_correct / self.num_sample

