Metrics
========================

User can use LPOT builtin metrics as well as register their own metrics.

## Builtin metric support list

LPOT supports some builtin metrics that popularly used in industry. Please refer to 'examples/helloworld/tf_example1' about how to config a builtin metric.

#### TensorFlow

| Type                  | Parameters        | Inputs          |
| :------               | :------           | :------         |
| topK                  | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   |
| Accuracy              | None              | preds, labels   |
| Loss                  | None              | preds, labels   |
| MAE                   | None              | preds, labels   |
| RMSE                  | None              | preds, labels   |
| MSE                   | None              | preds, labels   |
| F1                    | None              | preds, labels   |
| COCOmAP               | anno_path(str, default=None):annotation path | preds, labels   |
| BLEU                  | None              | preds, labels   |

#### PyTorch

| Type                  | Parameters        | Inputs          |
| :------               | :------           | :------         |
| topK                  | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   |
| Accuracy              | None              | preds, labels   |
| Loss                  | None              | preds, labels   |
| MAE                   | None              | preds, labels   |
| RMSE                  | None              | preds, labels   |
| MSE                   | None              | preds, labels   |
| F1                    | None              | preds, labels   |


#### MXNet

| Type                  | Parameters        | Inputs          |
| :------               | :------           | :------         |
| topK                  | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   |
| Accuracy              | None              | preds, labels   |
| Loss                  | None              | preds, labels   |
| MAE                   | None              | preds, labels   |
| RMSE                  | None              | preds, labels   |
| MSE                   | None              | preds, labels   |
| F1                    | None              | preds, labels   |



#### ONNXRT

| Type                  | Parameters        | Inputs          |
| :------               | :------           | :------         |
| topK                  | k (int, default=1): Number of top elements to look at for computing accuracy | preds, labels   |
| Accuracy              | None              | preds, labels   |
| Loss                  | None              | preds, labels   |
| MAE                   | None              | preds, labels   |
| RMSE                  | None              | preds, labels   |
| MSE                   | None              | preds, labels   |
| F1                    | None              | preds, labels   |


## User specific metric

User can register their own metric as follows:

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

After defining the metric class, user needs to encapsulate it into eval_func and pass eval_func to quantizer.

The pseudo code is as follows:

```python
def eval_func(graph):
    metric = Metric()
    for data, label in dataloader:
        output = sess.run(graph.output_tensor, feed_dict)
        metric.update(output, label)
    acc = metric.result()
    return acc

from lpot.quantization import Quantization
quantizer = Quantization(yaml_file)
q_model = quantizer(graph,
                    q_dataloader=dataloader,
                    eval_func=eval_func)

```
