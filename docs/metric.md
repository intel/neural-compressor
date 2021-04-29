Metrics
=======

In terms of evaluating the performance of a specific model, we should have general metrics to measure the performance of different models. Different frameworks always have their own Metric module but with different features and APIs. LPOT Metrics supports code-free configuration through a yaml file, with built-in metrics, so that LPOT can achieve performance and accuracy without code changes from the user. In special cases, users can also register their own metric classes through the LPOT method.

## How to use Metrics

### Config built-in metric in a yaml file

Users can specify an LPOT built-in metric such as shown below:

```yaml
evaluation:
  accuracy:
    metric:
      topk: 1
```


### Config custom metric in code

Users can also register their own metric as follows:

```python
class Metric(object):
    def __init__(self):
        # init code here

    def update(self, preds, labels):
        # add preds and labels to storage

    def reset(self):
        # clear preds and labels storage

    def result(self):
        # calculate accuracy
        return accuracy

```

The result() function returns a higher-is-better scalar to reflect model accuracy on an evaluation dataset.

After defining the metric class, users need to register it with a user-defined metric name and the metric class:

```python

from lpot.quantization import Quantization, common
quantizer = Quantization(yaml_file)
quantizer.model = common.Model(graph)
quantizer.metric = common.Metric(NewMetric, 'metric_name')
quantizer.calib_dataloader = dataloader
q_model = quantizer()

```

## Built-in metric support list

LPOT supports some built-in metrics that are popularly used in industry. 

Refer to [this HelloWorld example](/examples/helloworld/tf_example1) on how to config a built-in metric.

#### TensorFlow

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE()                 | None              | preds, labels   | Computes Mean Absolute Error (MAE) loss. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE()                | None              | preds, labels   | Computes Root Mean Squred Error (RMSE) loss. | metric: <br> &ensp;&ensp; RMSE: {} |
| MSE()                 | None              | preds, labels   | Computes Mean Squared Error (MSE) loss. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |
| COCOmAP(anno_path)    | anno_path(str, default=None):annotation path | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; COCOmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br> If anno_path is not set, metric will use built-in coco_label_map |
| BLEU()                | None              | preds, labels   |  BLEU score computation between labels and predictions. An approximate BLEU scoring method since we do not glue word pieces or decode the ids and tokenize the output. By default, we use ngram order of 4 and use brevity penalty. Also, this does not have beam search | metric: <br> &ensp;&ensp; BLEU: {} |
| SquadF1()             | None              | preds, labels   | Evaluate v1.1 of the SQuAD dataset | metric: <br> &ensp;&ensp; SquadF1: {} |


#### PyTorch

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Calculates the top-k categorical accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Calculates the accuracy for binary, multiclass and multilabel data. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.Accuracy) for details. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE()                 | None              | preds, labels   | Calculates the mean absolute error. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.MeanAbsoluteError) for details. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE()                | None              | preds, labels   | Calculates the root mean squared error. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.RootMeanSquaredError) for details. | metric: <br> &ensp;&ensp; RMSE: {} |
| MSE()                 | None              | preds, labels   | Calculates the mean squared error. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.MeanSquaredError) for details. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |

#### MXNet

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.Accuracy) for details. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE()                 | None              | preds, labels   | Computes Mean Absolute Error (MAE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MAE) for details. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE()                | None              | preds, labels   | Computes Root Mean Squred Error (RMSE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.RMSE) for details. | metric: <br> &ensp;&ensp; RMSE: {} |
| MSE()                 | None              | preds, labels   | Computes Mean Squared Error (MSE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MSE) for details. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.F1) for details. | metric: <br> &ensp;&ensp; F1: {} |


#### ONNXRT

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   |Computes accuracy classification score. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE()                 | None              | preds, labels   | Computes Mean Absolute Error (MAE) loss. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE()                | None              | preds, labels   | Computes Root Mean Squred Error (RMSE) loss.| metric: <br> &ensp;&ensp; RMSE: {} |
| MSE()                 | None              | preds, labels   | Computes Mean Squared Error (MSE) loss. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |