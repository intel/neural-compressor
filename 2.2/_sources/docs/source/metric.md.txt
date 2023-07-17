Metrics
=======
1. [Introduction](#introduction)
2. [Supported Built-in Metric Matrix](#supported-built-in-metric-matrix)

    2.1. [TensorFlow](#tensorflow)

    2.2. [PyTorch](#pytorch)
    
    2.3. [MxNet](#mxnet)

    2.4. [ONNXRT](#onnxrt)

3. [Get Started with Metric](#get-start-with-metric)
    
    3.1. [Use Intel® Neural Compressor Metric API](#use-intel®-neural-compressor-metric-api)
    
    3.2. [Build Custom Metric with Python API](#build-custom-metric-with-python-api)

## Introduction

In terms of evaluating the performance of a specific model, we should have general metrics to measure the performance of different models. Different frameworks always have their own Metric module but with different APIs and parameters. As for Intel® Neural Compressor, it implements an internal metric and provides a unified `Metric` API. In special cases, users can also register their own metric classes through [Build Custom Metric with Python API](#build-custom-metric-with-python-api).

## Supported Built-in Metric Matrix

Neural Compressor supports some built-in metrics that are popularly used in industry. 

### TensorFlow

| Metric                | Parameters        | Inputs          | Comments |
| :------               | :------           | :------         | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. |
| MAE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Mean Absolute Error (MAE) loss. |
| RMSE(compare_label)   | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Root Mean Square Error (RMSE) loss. |
| MSE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels | preds, labels   | Computes Mean Squared Error (MSE) loss. |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. |
| mAP(anno_path, iou_thrs, map_points)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str, default=0.5): Minimal value for intersection over union that allows to make decision that prediction bounding box is true positive. You can specify one float value between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int, default=0): The way to calculate mAP. 101 for 101-point interpolated AP, 11 for 11-point interpolated AP, 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| COCOmAP(anno_path, iou_thrs, map_points)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| VOCmAP(anno_path, iou_thrs, map_points)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs**(float or str): Intersection over union threshold. Set to 0.5.<br> **map_points**(int): The way to calculate mAP. The way to calculate mAP. Set to 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| COCOmAPv2(anno_path, iou_thrs, map_points, output_index_mapping)    | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br>**iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. <br> **output_index_mapping** (dict, default={'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}): Specifies the index of outputs in model raw prediction, -1 means this output does not exist. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| BLEU()                | None              | preds, labels   |  BLEU score computation between labels and predictions. An approximate BLEU scoring method since we do not glue word pieces or decode the ids and tokenize the output. By default, we use ngram order of 4 and use brevity penalty. Also, this does not have beam search |
| SquadF1()             | None              | preds, labels   | Evaluate v1.1 of the SQuAD dataset |


### PyTorch

| Metric                | Parameters        | Inputs          | Comments |
| :------               | :------           | :------         | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Calculates the top-k categorical accuracy. |
| Accuracy()            | None              | preds, labels   | Calculates the accuracy for binary, multiclass and multilabel data. <br> Please refer [Pytorch docs](https://pytorch.org/ignite/metrics.html#ignite.metrics.Accuracy) for details. |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. | 
| MAE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.               | preds, labels   | Computes Mean Absolute Error (MAE) loss. |
| RMSE(compare_label)   | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. |
| MSE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Mean Squared Error (MSE) loss. |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. |

### MXNet

| Metric                | Parameters        | Inputs          | Comments |
| :------               | :------           | :------         | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. |
| Accuracy()            | None              | preds, labels   | Computes accuracy classification score. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.Accuracy) for details. |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. |
| MAE()                 | None              | preds, labels   | Computes Mean Absolute Error (MAE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MAE) for details. |
| RMSE(compare_label)   | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. |
| MSE()                 | None              | preds, labels   | Computes Mean Squared Error (MSE) loss. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.MSE) for details. |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/metric/index.html#mxnet.metric.F1) for details. |


### ONNXRT

| Metric                | Parameters        | Inputs          | Comments |
| :------               | :------           | :------         | :------ |
| topk(k)               | **k** (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   | Computes top k predictions accuracy. |
| Accuracy()            | None              | preds, labels   |Computes accuracy classification score. |
| Loss()                | None              | preds, labels   | A dummy metric for directly printing loss, it calculates the average of predictions. <br> Please refer to [MXNet docs](https://mxnet.apache.org/versions/1.7.0/api/python/docs/_modules/mxnet/metric.html#Loss) for details. |
| MAE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.               | preds, labels   | Computes Mean Absolute Error (MAE) loss. |
| RMSE(compare_label)   | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Root Mean Squared Error (RMSE) loss. |
| MSE(compare_label)    | **compare_label** (bool, default=True): Whether to compare label. False if there are no labels and will use FP32 preds as labels.              | preds, labels   | Computes Mean Squared Error (MSE) loss. |
| F1()                  | None              | preds, labels   | Computes the F1 score of a binary classification problem. |
| mAP(anno_path, iou_thrs, map_points)     | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str, default=0.5): Minimal value for intersection over union that allows to make decision that prediction bounding box is true positive. You can specify one float value between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int, default=0): The way to calculate mAP. 101 for 101-point interpolated AP, 11 for 11-point interpolated AP, 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| COCOmAP(anno_path, iou_thrs, map_points) | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br> **iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| VOCmAP(anno_path, iou_thrs, map_points)  | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format .<br> **iou_thrs** (float or str): Intersection over union threshold. Set to 0.5.<br> **map_points**  (int): The way to calculate mAP. The way to calculate mAP. Set to 0 for area under PR curve. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| COCOmAPv2(anno_path, iou_thrs, map_points, output_index_mapping) | **anno_path** (str): Annotation path. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. The annotation file should be a yaml file, please refer to [label_map](../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq/label_map.yaml) for its format. <br>**iou_thrs** (float or str): Intersection over union threshold. Set to "0.5:0.05:0.95" for standard COCO thresholds.<br> **map_points** (int): The way to calculate mAP. Set to 101 for 101-point interpolated AP.<br> **output_index_mapping** (dict, default={'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}): Specifies the index of outputs in model raw prediction, -1 means this output does not exist. | preds, labels  | preds is a tuple which supports 2 length: 3 and 4. <br> If its length is 3, it should contain boxes, scores, classes in turn. <br> If its length is 4, it should contain target_boxes_num, boxes, scores, classes in turn <br> labels is a tuple which contains bbox, str_label, int_label, image_id inturn <br> the length of one of str_label and int_label can be 0 |
| GLUE(task)    | **task** (str, default=mrpc): The name of the task. Choices include mrpc, qqp, qnli, rte, sts-b, cola, mnli, wnli.              | preds, labels   | Computes GLUE score for bert model. |



## Get Started with Metric

### Use Intel® Neural Compressor Metric API

Users can specify a Neural Compressor built-in metric such as shown below:

```python
from neural_compressor import Metric
from neural_compressor import quantization, PostTrainingQuantConfig

top1 = Metric(name="topk", k=1)
config = PostTrainingQuantConfig()
q_model = fit(model, config, calib_dataloader=calib_dataloader, eval_dataloader=eval_dataloader,eval_metric=top1)
```

### Build Custom Metric with Python API

Please refer to [Metrics code](../neural_compressor/metric), users can also register their own metric as follows:

```python
class NewMetric(object):
    def __init__(self):
        # init code here

    def update(self, preds, labels):
        # add preds and labels to storage

    def reset(self):
        # clear preds and labels storage

    def result(self):
        # calculate metric result
        return result

```

The result() function returns a higher-is-better scalar to reflect model accuracy on an evaluation dataset.

After defining the metric class, users can initialize it and pass it to quantizer:

```python
from neural_compressor import quantization, PostTrainingQuantConfig

new_metric = NewMetric()
config = PostTrainingQuantConfig()
q_model = fit(model, config, calib_dataloader=calib_dataloader, eval_dataloader=eval_dataloader,eval_metric=new_metric)
```

## Example

- Refer to this [example](https://github.com/intel/neural-compressor/tree/master/examples/onnxrt/body_analysis/onnx_model_zoo/arcface/quantization/ptq_static) for how to define a customised metric.

- Refer to this [example](https://github.com/intel/neural-compressor/blob/master/examples/tensorflow/image_recognition/tensorflow_models/efficientnet-b0/quantization/ptq) for how to use internal metric.