Step-by-Step
============

This document is used to list the steps of reproducing quantization and benchmarking results.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
python run_clm.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --tune \
  --output_dir saved_results
```
> NOTE
>
> `saved_results` is the path to finetuned output_dir.

or
```bash
sh run_quant.sh --topology=topology_name --input_model=model_name_or_path
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --int8=true --input_model=model_name_or_path  --config=saved_results
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
  <tr>
    <td>gpt_j_wikitext</td>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6B">EleutherAI/gpt-j-6B</a></td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
  </tr>
  <tr>
    <td>gpt_neox</td>
    <td><a href="https://huggingface.co/abeja/gpt-neox-japanese-2.7b">abeja/gpt-neox-japanese-2.7b</a></td>
    <td><a href="https://huggingface.co/datasets/oscar">oscar</a></td>
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
For more details, please refer to the [sample code](./run_clm.py).
