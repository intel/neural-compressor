Step-by-Step
============

This document is used to list the steps of quantizing the model with `PostTrainingDynamic` on the translation task.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in requirements, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/fx
sh run_quant.sh --topology=topology_name --input_model=model_name_or_path
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --input_model=model_name_or_path --config=saved_results --int8=true
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
    <td>t5-small</td>
    <td><a href="https://huggingface.co/t5-small">t5-small</a></td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
  </tr>
  <tr>
    <td>marianmt_WMT_en_ro</td>
    <td><a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-ro">Helsinki-NLP/opus-mt-en-ro</a></td>
    <td><a href="https://huggingface.co/datasets/wmt16">wmt16</a></td>
  </tr>
</tbody>
</table>

## 4. Saving and Loading Model
### Saving model
```python
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig(approach="dynamic")
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
### Loading model
```python
from neural_compressor.utils.pytorch import load
quantized_model = load(tuned_checkpoint,
                       model)
```
--------
For more details, please refer to the [sample code](./run_translation.py).