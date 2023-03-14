Step-by-Step
============

This document is used to list the steps of reproducing quantization and benchmarking results.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/seq2seq/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/nlp/huggingface_models/summarization/quantization/ptq_dynamic/fx
pip install -r requirements.txt
```
## Prepare Dataset
```shell
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum.tar.gz
tar -xzvf billsum.tar.gz
```

# Run
## 1. Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/summarization/quantization/ptq_dynamic/fx
sh run_tuning.sh --topology=topology_name
```
> NOTE
>
> topology_name can be:{"pegasus_samsum"}
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --int8=true --config=saved_results
# fp32
sh run_benchmark.sh --topology=topology_name --mode=performance
```
## 3. Validated Model List
<table>
<thead>
  <tr>
    <th>Topology Name</th>
    <th>Model Name</th>
    <th>Dataset/Task Name</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>pegasus_samsum</td>
    <td><a href="https://huggingface.co/lvwerra/pegasus-samsum">lvwerra/pegasus-samsum</a></td>
    <td><a href="https://huggingface.co/datasets/samsum">samsum</a></td>
  </tr>
</tbody>
</table>

## 4. Saving and Loading Model
### Saving model:
```python
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
from neural_compressor import quantization
accuracy_criterion = AccuracyCriterion(higher_is_better=False, tolerable_loss=0.5)
conf = PostTrainingQuantConfig(accuracy_criterion=accuracy_criterion)
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=dataloader(),
                           eval_func=eval_func)
q_model.save("output_dir")
```
Here, `q_model` is the Neural Compressor model class, so it has "save" API:

```python
q_model.save("Path_to_save_quantized_model")
```
### Loading model:
```python
from neural_compressor.utils.pytorch import load
quantized_model = load(tuned_checkpoint,
                       model)
```
--------
For more details, please refer to the [sample code](./run_summarization.py).