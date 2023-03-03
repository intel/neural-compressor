Step-by-Step
============

This document is used to list the steps of reproducing quantization and benchmarking results.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/text-classification/README.md).

> **Note** Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
python run_glue.py \
        --model_name_or_path yoshitomo-matsubara/bert-large-uncased-rte \
        --task_name rte \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size 16 \
        --no_cuda \
        --output_dir /path/to/checkpoint/dir \
        --tune \
        --overwrite_output_dir
```
> NOTE: /path/to/checkpoint/dir is the path to finetune output_dir

or
```bash
sh run_tuning.sh --topology=topology_name --input_model=model_name_or_path
```
#### Try distributed tuning (take mrpc task as an example) as follows:

Install mpi4py: 

```shell
# Build Open MPI

# download openmpi-<version> from https://www.open-mpi.org/
tar xf openmpi-<version>.tar.bz2
cd openmpi-<version>
mkdir build
cd build
../configure --prefix=<path> 2>&1 | tee config.out
make all
make install

# add `Export PATH=<path>` into `~/.bashrc`
source ~/.bashrc

# Install mpi4py
pip install mpi4py
```

In `run_glue.py`, set `config.use_distributed_tuning` to True by the following statement.

```python
conf = PostTrainingQuantConfig(approach="static", tuning_criterion=tuning_criterion, use_distributed_tuning=True)
```

And then, run the following command:

```
mpirun -np <NUM_PROCESS> -mca btl_tcp_if_include <NETWORK_INTERFACE> -x OMP_NUM_THREADS=<MAX_NUM_THREADS> --host <HOSTNAME1>,<HOSTNAME2>,<HOSTNAME3> bash run_distributed_tuning.sh
```

* *`<NUM_PROCESS>`* is the number of processes, which is recommended to set to be equal to the number of hosts.

* *`<MAX_NUM_THREADS>`* is the number of threads, which is recommended to set to be equal to
the number of physical cores on one node.

* *`<HOSTNAME>`* is the host name, and argument `--host <HOSTNAME>,<HOSTNAME>...` can be replaced with `--hostfile <HOSTFILE>`, when each line in *`<HOSTFILE>`* is a host name.

* `-mca btl_tcp_if_include <NETWORK_INTERFACE>` is used to set the network communication interface between hosts. For example, *`<NETWORK_INTERFACE>`* can be set to 192.168.20.0/24 to allow the MPI communication between all hosts under the 192.168.20.* network segment.
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
    <td>bert_large_RTE</td>
    <td><a href="https://huggingface.co/yoshitomo-matsubara/bert-large-uncased-rte">yoshitomo-matsubara/bert-large-uncased-rte</a></td>
    <td>rte</a></td>
  </tr>
  <tr>
    <td>xlm-roberta-base_MRPC</td>
    <td><a href="https://huggingface.co/Intel/xlm-roberta-base-mrpc">Intel/xlm-roberta-base-mrpc</a></td>
    <td>mrpc</td>
  </tr>
</tbody>
</table>

# HuggingFace Model Hub
## 1. To upstream into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and int8 model weights in the format of [transformers](https://github.com/huggingface/transformers). 
```py
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into model hub and reuse them with our `OptimizedModel` API.

## 2. To download from HuggingFace model hub
We provide an API `OptimizedModel` to initialize int8 models from HuggingFace model hub and its usage same as the model class provided by [transformers](https://github.com/huggingface/transformers).
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
We also upstreamed several int8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.
----

# Tutorial of Enabling NLP Models with Intel® Neural Compressor
## 1. Intel® Neural Compressor supports two usages:
1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.
## 2. Code Prepare

We need to update `run_glue.py` like below:

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
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=eval_dataloader,
                           eval_func=eval_func)
q_model.save(training_args.output_dir)
```