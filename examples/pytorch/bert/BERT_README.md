Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT iLiT tuning zoo result.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> iLiT has no capability to solve this framework limitation. iLiT supposes user have done these two steps before invoking iLiT interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html
> The latest version of pytorch enabled INT8 layer_norm op, but the accuracy was regression. So you should tune BERT model on commit 24aac321718d58791c4e6b7cfa50788a124dae23.

# Prerequisite

### 1. Installation

  ```Shell
  # Install iLiT
  pip install ilit

  # Install PyTorch
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git reset --hard 24aac321718d58791c4e6b7cfa50788a124dae23
  git submodule sync
  git submodule update --init --recursive
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  python setup.py install
  
  # Install BERT model
  cd examples/bert
  python setup.py install

  # Install tensorboard
  pip install tensorboard

  # Install tqdm
  pip install tqdm

  # Install sklearn
  pip install scikit-learn
  ```

### 2. Prepare Dataset

  Before running any of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks)

### 3. Prepare pretrained model
  please refer to https://github.com/huggingface/transformers, and fine tune the model to get pretrained model.

# Run

### BERT

  ```Shell
  cd examples/pytorch/bert
  ./run_all.sh
  ```

Examples of enabling iLiT
=========================

This is a tutorial of how to enable BERT model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.

2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As BERT's matricses are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Write Yaml config file
In examples directory, there is conf.yaml. We could remove most of items and only keep mandotory item for tuning.
```
framework:
  - name: pytorch

device: cpu

tuning:
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.
> **Note** : iLiT tool don't support "mse" tuning strategy for pytorch framework

### prepare
PyTorch quantization requires two manual steps:

  1. Add QuantStub and DeQuantStub for all quantizable ops.
  2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu. In bert model, there is no fuse pattern.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.
The related code changes please refer to examples/pytorch/bert/transformers/modeling_bert.py.

### code update
After prepare step is done, we just need update run_squad_tune.py and run_glue_tune.py like below
```
class Bert_DataLoader(DataLoader):
    def __init__(self, loader=None, model_type=None, device='cpu'):
        self.loader = loader
        self.model_type = model_type
        self.device = device
    def __iter__(self):
        for batch in self.loader:
            batch = tuple(t.to(self.device) for t in batch)
            outputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if self.model_type != 'distilbert':
                outputs['token_type_ids'] = batch[2] if self.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            yield outputs, outputs['labels']
```

```
if args.do_ilit_tune:
    def eval_func_for_ilit(model):
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
    dataset = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=False)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    test_dataloader = Bert_DataLoader(eval_dataloader, args.model_type, args.device)
    import ilit
    tuner = ilit.Tuner("./conf.yaml")
    tuner.tune(model, test_dataloader, eval_func=eval_func_for_ilit)
    exit(0)
```
