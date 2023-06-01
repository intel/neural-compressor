Step-by-Step
============

This document is used to introduce the details about how to quantize the model with `PostTrainingStatic` on the text classification task and obtain the benchmarking results.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in requirements, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization

### 1.1 Quantization with single node

```shell
python run_glue.py \
        --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size 16 \
        --no_cuda \
        --output_dir saved_results \
        --tune \
        --overwrite_output_dir
```
> NOTE: `saved_results` is the path to finetuned output_dir.

or
```bash
bash run_tuning.sh --topology=topology_name --input_model=model_name_or_path
```


## 2. Benchmark
```bash
# int8
bash run_benchmark.sh --topology=topology_name --mode=performance --int8=true --input_model=saved_results
# fp32
bash run_benchmark.sh --topology=topology_name --mode=performance --input_model=model_name_or_path
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
    <td>distilbert_base_sst2_ipex</td>
    <td><a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>sst2</td>
  </tr>
  <tr>
    <td>distilbert_base_sst2_sq_ipex</td>
    <td><a href="https://huggingface.co/Intel/xlm-roberta-base-mrpc">distilbert-base-uncased-finetuned-sst-2-english</a></td>
    <td>sst2</td>
  </tr>
</tbody>
</table>
