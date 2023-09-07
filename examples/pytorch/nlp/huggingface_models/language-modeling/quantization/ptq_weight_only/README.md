Step-by-Step
============

This document is used to list the steps of reproducing weight only quantization and benchmarking results.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only
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
  --weight_only_bits 8 \
  --weight_only_group -1 \
  --weight_only_scheme sym \
  --weight_only_algorithm RTN \
  --tune \
  --output_dir saved_results
```
or
```bash
sh run_quant.sh --topology=gpt_j_wikitext_weight_only --input_model=EleutherAI/gpt-j-6B --weight_only_bits=8 --weight_only_group=-1 --weight_only_scheme=sym --weight_only_algorithm=RTN
```

> NOTE
>
> `weight_only_bits`, `weight_only_group`, `weight_only_scheme`, and `weight_only_algorithm` can be modified by user. For details, please refer to [README](../../../../../../../docs/source/quantization_weight_only.md).

### Run MLPerf on GPT-J-6B
Use the following link to get
[**CNN Daily Mail** datasets](https://github.com/intel-innersource/frameworks.ai.benchmarking.mlperf.submission.inference-submission-v3-1/tree/master/closed/Intel/code/gpt-j/pytorch-cpu#download-and-prepare-dataset)
and [gpt-j-6B mlperf model](https://github.com/mlcommons/inference/tree/master/language/gpt-j#download-gpt-j-model)

Then run following command to do quantization (please refer to *run_gptj_mlperf_int4.sh*)
```shell
python -u examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only/run_gptj_mlperf_int4.py \
    --model_name_or_path /your/gptj/model/ \
    --wbits 4 \
    --sym \
    --group_size -1 \
    --nsamples 128 \
    --calib-data-path /your/data/calibration-data/cnn_dailymail_calibration.json \
    --val-data-path /your/data/validation-data/cnn_dailymail_validation.json \
    --calib-iters 128 \
    --use_max_length \
    --use_fp16 \
    --use_gpu
```
Notes: for per channel quantization, set group_size to **-1**, otherwise 32, 128, etc. More comprehensive details about user-defined arguments are available at our [weight_onlly quantization documentations](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md#quantization-capability)

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
    <td>gpt_j_wikitext_weight_only</td>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6B">EleutherAI/gpt-j-6B</a></td>
    <td><a href="https://huggingface.co/datasets/wikitext">wikitext</a></td>
  </tr>
</tbody>
</table>

## 4. Saving and Loading Model
### Saving model:
```python
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
from neural_compressor import quantization
op_type_dict={
    '.*':{
        "weight": {
            'bits': 8,
            'group_size': 32,
            'scheme': 'sym', 
            'algorithm': 'RTN', 
        },
    },
}
accuracy_criterion = AccuracyCriterion(higher_is_better=False, tolerable_loss=0.01)
conf = PostTrainingQuantConfig(accuracy_criterion=accuracy_criterion,
                            approach='weight_only',
                            op_type_dict=op_type_dict)
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
quantized_model = load(tuned_checkpoint, model)
```
--------

For more details, please refer to the [sample code](./run_clm.py).

# (May Remove Later) Run GPTQ algorithm
```
sh run-gptq-llm.sh
# You may want to move script run-gptq-llm.sh to root dir of neural compressor and modify python file's path.
# Please make sure pile dataset is downloaded.
```
