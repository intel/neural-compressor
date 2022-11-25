Metrics
=======
1. [Introduction](#introduction)
2. [Supported Built-in Metric Matrix](#supported-built-in-metric-matrix)

    2.1. [TensorFlow](#tensorflow)

    2.2. [PyTorch](#pytorch)
    
    2.3. [MxNet](#mxnet)

    2.4. [ONNXRT](#onnxrt)

3. [Get Start with Metrics](#get-start-with-metrics)
    
    3.1. [Support Single-metric and Multi-metrics](#support-single-metric-and-multi-metrics)
    
    3.2. [Build Custom Metric with Python API](#build-custom-metric-with-python-api)

## Introduction

In terms of evaluating the performance of a specific model, we should have general metrics to measure the performance of different models. Different frameworks always have their own Metric module but with different APIs and parameters. Neural Compressor Metrics supports code-free configuration through a yaml file, with built-in metrics, so that Neural Compressor can achieve performance and accuracy without code changes from the user. In special cases, users can also register their own metric classes through [building custom metric in code](#build-custom-metric-in-code).

## Supported Built-in Metric Matrix

Neural Compressor supports some built-in metrics that are popularly used in industry. 

### TensorFlow

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Mean Absolute Error (MAE) loss. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE(compare_label)   | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Root Mean Square Error (RMSE) loss. | metric: <br> &ensp;&ensp; RMSE: {} |
| MSE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Mean Squared Error (MSE) loss. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |
| mAP(anno_path, iou_thrs, map_points)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str, default=0.5): Minimal value for intersection over union that allows to make decision that prediction bounding box is true positive. You can specify one float value between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int, default=0): The way to calculate mAP. 101 for 101-point interpolated AP, 11 for 11-point interpolated AP, 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; mAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br> &ensp;&ensp;&ensp;&ensp; iou_thrs: 0.5 <br> &ensp;&ensp;&ensp;&ensp; map_points: 0 <br><br> If anno_path is not set, metric will use official coco label id |
| COCOmAP(anno_path, iou_thrs, map_points)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; COCOmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br><br> If anno_path is not set, metric will use official coco label id |
| VOCmAP(anno_path, iou_thrs, map_points)    | **anno_path**(str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs**(float or str): Intersection over union threshold. Set to 0.5.<br> **map_points**(int): The way to calculate mAP. The way to calculate mAP. Set to 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; VOCmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br><br> If anno_path is not set, metric will use official coco label id |
| COCOmAPv2(anno_path, iou_thrs, map_points, output_index_mapping)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br>**iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. <br> **output_index_mapping**(dict, default={'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}): Specifies the index of outputs in model raw prediction, -1 means this output does not exist. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; COCOmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br> &ensp;&ensp;&ensp;&ensp; output_index_mapping: <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; num_detections: 0 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; boxes: 1 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; scores: 2 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; classes: 3 <br><br> If anno_path is not set, metric will use official coco label id |
| BLEU()                | None              | preds, labels   |  BLEU score computation between labels and predictions. An approximate BLEU scoring method since we do not glue word pieces or decode the ids and tokenize the output. By default, we use ngram order of 4 and use brevity penalty. Also, this does not have beam search | metric: <br> &ensp;&ensp; BLEU: {} |
| SquadF1()             | None              | preds, labels   | Evaluate v1.1 of the SQuAD dataset | metric: <br> &ensp;&ensp; SquadF1: {} |


### PyTorch

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Calculates the top-k categorical accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Calculates the accuracy for binary, multiclass and multilabel data. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.Accuracy) for details. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE(compare_label)    | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.               | preds, labels   | Computes Mean Absolute Error (MAE) loss. | metric: <br> &ensp;&ensp; MAE: <br> &ensp;&ensp;&ensp;&ensp; compare_label： True |
| RMSE(compare_label)   | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. | metric: <br> &ensp;&ensp; RMSE: <br> &ensp;&ensp;&ensp;&ensp; compare_label: True |
| MSE(compare_label)    | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Mean Squared Error (MSE) loss. | metric: <br> &ensp;&ensp; MSE: <br> &ensp;&ensp;&ensp;&ensp; compare_label: True |(https://pytorch.org/ignite/metrics.html#ignite.metrics.MeanSquaredError) for details. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |

### MXNet

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.Accuracy) for details. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE()                 | None              | preds, labels   | Computes Mean Absolute Error (MAE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MAE) for details. | metric: <br> &ensp;&ensp; MAE: {} |
| RMSE(compare_label)   | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. | metric: <br> &ensp;&ensp; RMSE: <br> &ensp;&ensp;&ensp;&ensp; compare_label: True |
| MSE()                 | None              | preds, labels   | Computes Mean Squared Error (MSE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MSE) for details. | metric: <br> &ensp;&ensp; MSE: {} |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.F1) for details. | metric: <br> &ensp;&ensp; F1: {} |


### ONNXRT

| Metric                | Parameters        | Inputs          | Comments | Usage(In yaml file) |
| :------               | :------           | :------         | :------ | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. | metric: <br> &ensp;&ensp; topk: <br> &ensp;&ensp;&ensp;&ensp; k: 1 |
| Accuracy()            | None              | preds, labels   |Computes accuracy classification score. | metric: <br> &ensp;&ensp; Accuracy: {} |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | metric: <br> &ensp;&ensp; Loss: {} |
| MAE(compare_label)    | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.               | preds, labels   | Computes Mean Absolute Error (MAE) loss. | metric: <br> &ensp;&ensp; MAE: <br> &ensp;&ensp;&ensp;&ensp; compare_label： True |
| RMSE(compare_label)   | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. | metric: <br> &ensp;&ensp; RMSE: <br> &ensp;&ensp;&ensp;&ensp; compare_label: True |
| MSE(compare_label)    | **compare_label**(bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Mean Squared Error (MSE) loss. | metric: <br> &ensp;&ensp; MSE: <br> &ensp;&ensp;&ensp;&ensp; compare_label: True |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. | metric: <br> &ensp;&ensp; F1: {} |
| mAP(anno_path, iou_thrs, map_points)     | **anno_path**(str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs**(float or str, default=0.5): Minimal value for intersection over union that allows to make decision that prediction bounding box is true positive. You can specify one float value between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.<br> **map_points**(int, default=0): The way to calculate mAP. 101 for 101-point interpolated AP, 11 for 11-point interpolated AP, 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; mAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br> &ensp;&ensp;&ensp;&ensp; iou_thrs: 0.5 <br> &ensp;&ensp;&ensp;&ensp; map_points: 0 <br><br> If anno_path is not set, metric will use official coco label id |
| COCOmAP(anno_path, iou_thrs, map_points) | **anno_path**(str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs**(float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points**(int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; COCOmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br><br> If anno_path is not set, metric will use official coco label id |
| VOCmAP(anno_path, iou_thrs, map_points)  | **anno_path**(str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format .<br> **iou_thrs**(float or str): Intersection over union threshold. Set to 0.5.<br> **map_points**(int): The way to calculate mAP. The way to calculate mAP. Set to 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; VOCmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br><br> If anno_path is not set, metric will use official coco label id |
| COCOmAPv2(anno_path, iou_thrs, map_points, output_index_mapping) | **anno_path**(str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br>**iou_thrs**(float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points**(int): The way to calculate mAP. Set to 101 for 101-point interpolated AP.<br> **output_index_mapping**(dict, default={'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}): Specifies the index of outputs in model raw prediction, -1 means this output does not exist. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 | metric: <br> &ensp;&ensp; COCOmAP: <br> &ensp;&ensp;&ensp;&ensp; anno_path: /path/to/annotation <br> &ensp;&ensp;&ensp;&ensp; output_index_mapping: <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; num_detections: 0 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; boxes: 1 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; scores: 2 <br> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; classes: 3<br><br> If anno_path is not set, metric will use official coco label id |
| GLUE(task)    | **task** (str, default=mrpc): The name of the task. Choices include mrpc, qqp, qnli, rte, sts-b, cola, mnli, wnli.              | preds, labels   | Computes GLUE score for bert model. | metric: <br> &ensp;&ensp; GLUE: <br> &ensp;&ensp;&ensp;&ensp; task: mrpc |


## Get Start with Metrics

### Support Single-metric and Multi-metrics
Users can specify an Neural Compressor built-in metric such as shown below:

```yaml
evaluation:
  accuracy:
    metric:
      topk: 1
```

In some cases, users want to use more than one metric to evaluate the performance of a specific model and they can realize it with multi_metrics of Neural Compressor. Currently multi_metrics supports built-in metrics.


There are two usages for multi_metrics of Neural Compressor:

1. Evaluate performance of a model with metrics one by one

```yaml
evaluation:
  accuracy:
    multi_metrics:
      topk: 1
      MSE:
        compare_label: False
      higher_is_better: [True, False] # length of higher_is_better should be equal to num of metric, default is True
```

2. Evaluate performance of a model with weighted metric results

```yaml
evaluation:
  accuracy:
    multi_metrics:
      topk: 1
      MSE:
        compare_label: False
      weight: [0.5, 0.5] # length of weight should be equal to num of metric
      higher_is_better: [True, False] # length of higher_is_better should be equal to num of metric, default is True
```


### Build Custom Metric with Python API

Please refer to [Metrics code](../neural_compressor/experimental/metric), users can also register their own metric as follows:

```python
class NewMetric(object):
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

After defining the metric class, users can initialize it and pass it to quantizer:

```python

from neural_compressor.quantization import Quantization
quantizer = Quantization(yaml_file)
quantizer.model = graph
quantizer.metric = NewMetric()
quantizer.calib_dataloader = dataloader
q_model = quantizer.fit()

```
