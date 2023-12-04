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
        --model_name_or_path yoshitomo-matsubara/bert-large-uncased-rte \
        --task_name rte \
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
bash run_quant.sh --topology=topology_name --input_model=model_name_or_path
```

### 1.2 Quantization with multi-node

- Prerequisites:
    - [Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build)
    - [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html#using-pip)


> NOTE: User can also install Open MPI with [Conda](https://anaconda.org/conda-forge/openmpi).


In `run_glue.py`, set `config.quant_leve1` to 1 and `config.tuning_criterion.strategy` to "basic" by the following statement.

```python
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
tuning_criterion = TuningCriterion(max_trials=600, strategy="basic")
conf = PostTrainingQuantConfig(quant_level=1, approach="static", tuning_criterion=tuning_criterion)
```

And then, run the following command:

``` shell
mpirun -np <NUM_PROCESS> \
         -mca btl_tcp_if_include <NETWORK_INTERFACE> \
         -x OMP_NUM_THREADS=<MAX_NUM_THREADS> \
         --host <HOSTNAME1>,<HOSTNAME2>,<HOSTNAME3> \
         bash run_distributed_tuning.sh
```

* *`<NUM_PROCESS>`* is the number of processes, which is recommended to set to be equal to the number of hosts.

* *`<MAX_NUM_THREADS>`* is the number of threads, which is recommended to set to be equal to
the number of physical cores on one node.

* *`<HOSTNAME>`* is the host name, and argument `--host <HOSTNAME>,<HOSTNAME>...` can be replaced with `--hostfile <HOSTFILE>`, when each line in *`<HOSTFILE>`* is a host name.

* `-mca btl_tcp_if_include <NETWORK_INTERFACE>` is used to set the network communication interface between hosts. For example, *`<NETWORK_INTERFACE>`* can be set to 192.168.20.0/24 to allow the MPI communication between all hosts under the 192.168.20.* network segment.


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
    <td>bert_large_RTE</td>
    <td><a href="https://huggingface.co/yoshitomo-matsubara/bert-large-uncased-rte">yoshitomo-matsubara/bert-large-uncased-rte</a></td>
    <td>rte</td>
  </tr>
  <tr>
    <td>xlm-roberta-base_MRPC</td>
    <td><a href="https://huggingface.co/Intel/xlm-roberta-base-mrpc">Intel/xlm-roberta-base-mrpc</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>bert_base_MRPC</td>
    <td><a href="https://huggingface.co/Intel/bert-base-uncased-mrpc">Intel/bert-base-uncased-mrpc</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>bert_base_CoLA</td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-CoLA">textattack/bert-base-uncased-CoLA</a></td>
    <td>cola</td>
  </tr>
  <tr>
    <td>bert_base_STS-B</td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-STS-B">textattack/bert-base-uncased-STS-B</a></td>
    <td>stsb</td>
  </tr>
  <tr>
    <td>bert_base_SST-2</td>
    <td><a href="https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2">gchhablani/bert-base-cased-finetuned-sst2</a></td>
    <td>sst2</td>
  </tr>
  <tr>
    <td>bert_base_RTE</td>
    <td><a href="https://huggingface.co/ModelTC/bert-base-uncased-rte">ModelTC/bert-base-uncased-rte</a></td>
    <td>rte</td>
  </tr>
  <tr>
    <td>bert_large_QNLI</td>
    <td><a href="https://huggingface.co/textattack/bert-base-uncased-QNLI">textattack/bert-base-uncased-QNLI</a></td>
    <td>qnli</td>
  </tr>
  <tr>
    <td>bert_large_CoLA</td>
    <td><a href="https://huggingface.co/yoshitomo-matsubara/bert-large-uncased-cola">yoshitomo-matsubara/bert-large-uncased-cola</a></td>
    <td>cola</td>
  </tr>
  <tr>
    <td>distilbert_base_MRPC</td>
    <td><a href="https://huggingface.co/textattack/distilbert-base-uncased-MRPC">textattack/distilbert-base-uncased-MRPC</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>xlnet_base_cased_MRPC</td>
    <td><a href="https://huggingface.co/Intel/xlnet-base-cased-mrpc">Intel/xlnet-base-cased-mrpc</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>roberta_base_MRPC</td>
    <td><a href="https://huggingface.co/textattack/roberta-base-MRPC">textattack/roberta-base-MRPC</a></td>
    <td>mnli</td>
  </tr>
  <tr>
    <td>camembert_base_MRPC</td>
    <td><a href="https://huggingface.co/Intel/camembert-base-mrpc">Intel/camembert-base-mrpc</a></td>
    <td>mrpc</td>
  </tr>
</tbody>
</table>

# HuggingFace Model Hub
## 1. To upstream into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and INT8 model weights in the format of [transformers](https://github.com/huggingface/transformers).
```python
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into the model hub and reuse them with our `OptimizedModel` API.

## 2. To download from HuggingFace model hub
We provide an API `OptimizedModel` to initialize INT8 models from HuggingFace model hub and its usage is same as the model class provided by [transformers](https://github.com/huggingface/transformers).
```python
from neural_compressor.utils.load_huggingface import OptimizedModel
model = OptimizedModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
```
We also upstreamed several INT8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.
----

# Tutorial of Enabling NLP Models with Intel® Neural Compressor
## 1. Intel® Neural Compressor supports two usages:
1. User specifies FP32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies FP32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

## 2. Code Prepare
We update `run_glue.py` as follows:

```python
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

eval_dataloader = trainer.get_eval_dataloader()
batch_size = eval_dataloader.batch_size
metric_name = "eval_f1"

def take_eval_steps(model, trainer, metric_name, save_metrics=False):
    trainer.model = model
    metrics = trainer.evaluate()
    return metrics.get(metric_name)

def eval_func(model):
    return take_eval_steps(model, trainer, metric_name)

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(approach="static", tuning_criterion=tuning_criterion, use_distributed_tuning=False)
q_model = fit(model, conf=conf, calib_dataloader=eval_dataloader, eval_func=eval_func)
q_model.save(training_args.output_dir)
```
