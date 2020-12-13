Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.

> **Note**
>
> 1. PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Low Precision Optimization Tool has no capability to solve this framework limitation. Intel® Low Precision Optimization Tool supposes user have done these two steps before invoking Intel® Low Precision Optimization Tool interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html
> 2. The latest version of pytorch enabled INT8 layer_norm op, but the accuracy was regression. So you should tune BERT model on commit 24aac321718d58791c4e6b7cfa50788a124dae23.

# Prerequisite

### 1. Installation
  #### Python First
  Recommend python 3.6 or higher version.

  #### Install dependency
  ```
  pip install -r requirements.txt
  ```

  #### Install PyTorch
  You will need a C++14 compiler. Also, we highly recommend installing an Anaconda environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro. 
  ```Shell
  # Install Dependencies
  conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
  # Install pytorch from source
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git reset --hard 24aac321718d58791c4e6b7cfa50788a124dae23
  git submodule sync
  git submodule update --init --recursive
  python setup.py install
  ```

  #### Install BERT model
  ```Shell
  cd examples/pytorch/language_translation
  python setup.py install
  ```
  > **Note**
  >
  > Please don't install public transformers package.


### 2. Prepare Dataset

   * Before running any of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory `$GLUE_DIR`.
   * For SQuAD task, you should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/)

### 3. Prepare pretrained model
  Before use Intel® Low Precision Optimization Tool, you should fine tune the model to get pretrained model, You should also install the additional packages required by the examples:

  ```shell
  cd examples/pytorch/language_translation
  pip install -r examples/requirements.txt
  ```

   * For BERT base and glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...)  

  ```shell
  export GLUE_DIR=/path/to/glue
  export TASK_NAME=MRPC
  
  cd examples/pytorch/language_translation
  python examples/run_glue_tune.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK_NAME \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=8   \
      --per_gpu_train_batch_size=8   \
      --learning_rate 2e-5 \
      --num_train_epochs 3.0 \
      --output_dir /path/to/checkpoint/dir
  ```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

please refer to [BERT base scripts and instructions](README.md#run_gluepy-fine-tuning-on-glue-tasks-for-sequence-classification).

   * For BERT large and glue tasks(MRPC, CoLA, RTE, QNLI...)

  ```bash
  export GLUE_DIR=/path/to/glue
  export TASK_NAME=MRPC
  cd examples/pytorch/language_translation
  python -m torch.distributed.launch examples/run_glue_tune.py   \
      --model_type bert \
      --model_name_or_path bert-large-uncased-whole-word-masking \
      --task_name MRPC \
      --do_train   \
      --do_eval   \
      --do_lower_case   \
      --data_dir $GLUE_DIR/MRPC/   \
      --max_seq_length 128   \
      --per_gpu_eval_batch_size=8   \
      --per_gpu_train_batch_size=8   \
      --learning_rate 2e-5   \
      --num_train_epochs 3.0  \
      --output_dir /path/to/checkpoint/dir \
      --overwrite_output_dir   \
      --overwrite_cache \
  ```
This example code fine-tunes the Bert Whole Word Masking model on the Microsoft Research Paraphrase Corpus (MRPC) corpus using distributed training on 8 V100 GPUs to reach a F1 > 92.
Training with these hyper-parameters gave us the following results:

  ```bash
    acc = 0.8823529411764706
    acc_and_f1 = 0.901702786377709
    eval_loss = 0.3418912578906332
    f1 = 0.9210526315789473
    global_step = 174
    loss = 0.07231863956341798
  ```

please refer to [BERT large scripts and instructions](README.md#fine-tuning-bert-model-on-the-mrpc-classification-task)

   * For BERT large SQuAD task
  ```bash
  cd examples/pytorch/language_translation
  python -m torch.distributed.launch examples/run_squad.py \
      --model_type bert \
      --model_name_or_path bert-large-uncased-whole-word-masking \
      --do_train \
      --do_eval \
      --do_lower_case \
      --train_file $SQUAD_DIR/train-v1.1.json \
      --predict_file $SQUAD_DIR/dev-v1.1.json \
      --learning_rate 3e-5 \
      --num_train_epochs 2 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --output_dir /path/to/checkpoint/dir \
      --per_gpu_eval_batch_size=3   \
      --per_gpu_train_batch_size=3   \
  ```
Training with these hyper-parameters gave us the following results:

  ```bash
  python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
  {"exact_match": 86.91579943235573, "f1": 93.1532499015869}
  ```

please refer to [BERT large SQuAD instructions](README.md#run_squadpy-fine-tuning-on-squad-for-question-answering)

  * After fine tuning, you can get a checkpoint dir which include pretrained model, tokenizer and training arguments. This checkpoint dir will be used by lpot tuning as below.

# Run

### BERT glue task

  ```Shell
  export GLUE_DIR=/path/to/glue
  export TASK_NAME=MRPC

  cd examples/pytorch/language_translation
  python examples/run_glue_tune.py \
      --model_type bert \
      --model_name_or_path /path/to/checkpoint/dir \
      --task_name $TASK_NAME \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK_NAME \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size 8 \
      --no_cuda \
      --tune \
      --output_dir /path/to/checkpoint/dir
  ```
  where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.
  Where output_dir is path of checkpoint which be created by fine tuning.


### BERT SQuAD

  ```Shell
  cd examples/pytorch/language_translation

  python examples/run_squad_tune.py \
      --model_type bert \
      --model_name_or_path /path/to/checkpoint/dir \
      --task_name "SQuAD" \
      --do_eval \
      --data_dir /path/to/SQuAD/dataset \
      --max_seq_length 384 \
      --per_gpu_eval_batch_size 16 \
      --no_cuda \
      --tune \
      --output_dir /path/to/checkpoint/dir
  ```
  Where output_dir is path of checkpoint which be created by fine tuning.


Examples of enabling Intel® Low Precision Optimization Tool
=========================

This is a tutorial of how to enable BERT model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.

2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As BERT's matricses are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Write Yaml config file
In examples directory, there is conf.yaml. We could remove most of items and only keep mandotory item for tuning.
```
model:
  name: bert
  framework: pytorch

device: cpu

tuning:
    accuracy_criterion:
      relative: 0.01
    exit_policy:
      timeout: 0
      max_trials: 300
    random_seed: 9527
```
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.
> **Note** : lpot does NOT support "mse" tuning strategy for pytorch framework

### prepare
PyTorch quantization requires two manual steps:

  1. Add QuantStub and DeQuantStub for all quantizable ops.
  2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu. In bert model, there is no fuse pattern.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.
The related code changes please refer to examples/pytorch/bert/transformers/modeling_bert.py.

### code update
After prepare step is done, we just need update run_squad_tune.py and run_glue_tune.py like below
```
if args.tune:
    def eval_func_for_lpot(model):
        result, _ = evaluate(args, model, tokenizer)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        bert_task_acc_keys = ['best_f1', 'f1', 'mcc', 'spearmanr', 'acc']
        for key in bert_task_acc_keys:
            if key in result.keys():
                logger.info("Finally Eval {}:{}".format(key, result[key]))
                acc = result[key]
                break
        return acc
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=False)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    from lpot import Quantization
    quantizer = Quantization("./conf.yaml")
    if eval_task != "squad":
        eval_task = 'classifier'
    eval_dataset = quantizer.dataset('bert', dataset=eval_dataset,
                                     task=eval_task, model_type=args.model_type)
    test_dataloader = quantizer.dataloader(eval_dataset, batch_size=args.eval_batch_size)
    quantizer(model, test_dataloader, eval_func=eval_func_for_lpot)
    exit(0)
```

# Original BERT README
Please refer [BERT README](BERT_README.md)
