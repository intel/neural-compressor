Step-by-Step
============

This document is used to list the steps of reproducing PyTorch BERT tuning zoo result.

# Prerequisite

## 1. Environment

The dependent packages are all in requirements, please install as following.

```bash
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
python run_clm.py \
  --model_name_or_path google/reformer-crime-and-punishment \
  --dataset_name crime_and_punish\
  --do_train \
  --do_eval \
  --tune \
  --overwrite_output_dir \
  --output_dir /path/to/checkpoint/dir
```
or
```bash
sh run_tuning.sh --topology=topology_name --input_model=model_name_or_path
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --int8=true --input_model=/path/to/checkpoint/dir
# fp32
sh run_benchmark.sh --topology=topology_name --mode=performance --input_model=model_name_or_path
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
    <td>reformer_crime_and_punishment</td>
    <td><a href="https://huggingface.co/google/reformer-crime-and-punishment">google/reformer-crime-and-punishment</a></td>
    <td>crime_and_punish</td>
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
For more details, please refer to the [Sample code](./run_clm.py).
