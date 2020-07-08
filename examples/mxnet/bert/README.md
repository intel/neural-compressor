Step-by-Step
============

This document is used to list steps of reproducing MXNet BERT_base MRPC/Squad iLiT tuning zoo result.



# Prerequisite

### 1. Installation

  ```Shell
  # Install iLiT
  pip install ilit

  # Install MXNet
  pip install mxnet-mkl==1.6.0
  
  # Install gluonnlp
  pip install gluonnlp

  ```

### 2. Prepare Dataset

  Download GLUE dataset to work dir, naming as ~/.mxnet/datasets/glue_mrpc/.


# Run

### bert_base MRPC
```
 python3 finetune_classifier.py \
        --task_name MRPC \
        --only_inference \
        --model_parameters ./output_dir/model_bert_MRPC_4.params \
        --ilit_tune

```

### bert_base Squad
```
python3 finetune_squad.py \
        --model_parameters ./output_dir/net.params \
        --round_to 128 \
        --test_batch_size 128 \
        --only_predict \
        --ilit_tune
```
 

Examples of enabling iLiT auto tuning on MXNet BERT_base
=======================================================

This is a tutorial of how to enable a MXNet BERT base model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

We integrate MXNet BERT_base MRPC/Squad with iLiT by the second use case.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
#conf.yaml

framework:
  - name: mxnet

tuning:
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```

Because we use the second use case which need user to provide a custom "eval_func" encapsulates the evaluation dataset and metric, so we can not see a metric at config file tuning filed. We set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.


### code update
First, we need to construct evaluate function for ilit. At eval_func, we get the dev_data_list for the origin script, and return acc metric to ilit.

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
    # iLiT auto-tuning
    calib_data = dev_data_list[0][1]
    import ilit
    bert_tuner = ilit.Tuner("./bert.yaml")
    bert_tuner.tune(model, q_dataloader=calib_data, val_dataloader=calib_data, eval_func=test_func)

```

The iLiT tune() function will return a best quantized model during timeout constrain.
