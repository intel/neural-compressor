Step-by-Step
============

This document is used to list steps of reproducing MXNet BERT_base MRPC/Squad tuning zoo result.



# Prerequisite

### 1. Installation

  ```Shell
  # Install Intel® Low Precision Optimization Tool
  pip install lpot

  # Install MXNet
  pip install mxnet-mkl==1.6.0
  
  # Install gluonnlp
  pip install gluonnlp

  ```

### 2. Dataset

  The script `finetune_classifier.py` will download GLUE dataset automaticly to the directory **~/.mxnet/datasets/glue_mrpc/**, for more GLUE dataset informations, see [here](https://github.com/dmlc/gluon-nlp/blob/5dc6b9c9fab9e99b155554a50466c514b879ea84/src/gluonnlp/data/glue.py#L590).

  The script `finetune_squad.py` will download SQuAD dataset automaticly to the directory **~/.mxnet/datasets/squad/**, for more SQuAD dataset informations, see [here](https://github.com/dmlc/gluon-nlp/blob/5dc6b9c9fab9e99b155554a50466c514b879ea84/src/gluonnlp/data/question_answering.py#L36).


### 3. Finetune model
  - BERT_base need to do finetune to get a finetuned model for specific task. For MRPC, you need to run below command to get a finetuned model. After this, you can get a finetuned model at directory **./output_dir**, named as **model_bert_MRPC_4.params**.

  ```bash
  python3 finetune_classifier.py --batch_size 32 --lr 2e-5 --epochs 5 --seed 27 --task_name MRPC --warmup_ratio 0.1
  ```
   

  - For SQuAD task, you need to run below command to get a finetuned model. After this, you can get a finetuned model at directory **./output_dir**, named as **net.params**.
  ```bash
  python finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2
  ```
  

  >More informations for BERT finetune, please see [here](https://github.com/dmlc/gluon-nlp/blob/5dc6b9c9fab9e99b155554a50466c514b879ea84/scripts/bert/index.rst#sentence-classification).
# Run

### bert_base MRPC
```
 python3 finetune_classifier.py \
        --task_name MRPC \
        --bert_model bert_12_768_12 \
        --only_inference \
        --model_parameters ./output_dir/model_bert_MRPC_4.params \
        --tune
```

### bert_base Squad
```
python3 finetune_squad.py \
        --model_parameters ./output_dir/net.params \
        --round_to 128 \
        --test_batch_size 128 \
        --only_predict \
        --tune
```
 

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on MXNet BERT_base
=======================================================

This is a tutorial of how to enable a MXNet BERT base model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

We integrate MXNet BERT_base MRPC/Squad with Intel® Low Precision Optimization Tool by the second use case.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
# conf.yaml

model:                                  
  name: bert 
  framework: mxnet

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527

```

Because we use the second use case which need user to provide a custom "eval_func" encapsulates the evaluation dataset and metric, so we can not see a metric at config file tuning filed. We set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.


### code update
First, we need to construct evaluate function for lpot. At eval_func, we get the dev_data_list for the origin script, and return acc metric to lpot.

```python
    # define test_func
    def test_func(graph):

        graph.hybridize(static_alloc=True, static_shape=True)
        for segment, dev_data in dev_data_list:
            metric_nm, metric_val = evaluate(model=graph, 
                                            loader_dev=dev_data, 
                                            metric=task.metrics, 
                                            segment=segment)
        acc = metric_val[0]
        F1 = metric_val[1]

        return acc
```

After prepare step is done, we just need update main.py like below.

```python
    # Intel® Low Precision Optimization Tool auto-tuning
    calib_data = dev_data_list[0][1]
    from lpot import Quantization
    quantizer = Quantization("./bert.yaml")
    quantizer(model, q_dataloader=calib_data, val_dataloader=calib_data, eval_func=test_func)

```

The quantizer() function will return a best quantized model during timeout constrain.
