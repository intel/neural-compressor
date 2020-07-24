Step-by-Step
============

This document describes the step-by-step instructions for reproducing MXNet SSD-ResNet50_v1/SSD-Mobilenet 1.0 tuning results with iLiT.



# Prerequisite
### 1. Installation

  ```Shell
  # Install iLiT
  pip install ilit

  # Install MXNet
  pip install mxnet-mkl==1.6.0

  # Install gluoncv
  pip install gluoncv

  # Install pycocotool
  pip install pycocotools

  ```

### 2. Prepare Dataset

If you want to use VOC2007 dataset, download [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) Raw image to the directory **~/.mxnet/datasets/voc** (Note:this path is unchangeable per original inference script requirement)

If you want to use COCO2017 dataset, download [COCO2017](https://cocodataset.org/#download) Raw image to the directory **~/.mxnet/datasets/coco** (Note:this path is unchangeable per original inference script requirement)

# Run

### SSD-ResNet50_v1-VOC
```bash
python eval_ssd.py --network=resnet50_v1 --data-shape=512 --batch-size=256 --dataset voc --ilit_tune
```

### SSD-Mobilenet1.0-VOC
```bash
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=32 --dataset voc --ilit_tune
```

### SSD-ResNet50_v1-COCO
```bash
python eval_ssd.py --network=resnet50_v1 --data-shape=512 --batch-size=256 --dataset coco --ilit_tune
```

### SSD-Mobilenet1.0-COOC
```bash
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=32 --dataset coco --ilit_tune
```

Examples of enabling iLiT auto tuning on MXNet Object detection
=======================================================

This is a tutorial of how to enable a MXNet Object detection model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As this example use VOC/COCO dataset, use VOCMApMetrics/COCOEval as metric which is can find [here](https://github.com/dmlc/gluon-cv/blob/20a2ed3942720550728ce36c2be53b2d5bbbb6fd/gluoncv/utils/metrics/voc_detection.py#L13) and [here](https://cocodataset.org/). So we integrate MXNet SSD-ResNet50_v1/SSD-Mobilenet1.0 with iLiT by the second use case.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandatory items for tuning.


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

Because we use the second use case which need user to provide a custom "eval_func" which encapsulates the evaluation dataset and metric, we can not see a metric at config file tuning filed. We set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.


### code update

First, we need to construct evaluate function for iLiT. At eval_func, we get the val_dataset for the origin script, and return mAP metric to iLiT.

```python
    # define test_func
    def eval_func(graph):
        val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
        val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
        classes = val_dataset.classes  # class names

        size = len(val_dataset)
        ctx = [mx.cpu()]
        results = validate(graph, val_data, ctx, classes, size, val_metric)

        mAP = float(results[-1][-1])

        return mAP
```

After preparation is done, we just need update main.py like below.

```python

    # Doing iLiT auto-tuning here
    import ilit
    ssd_tuner = ilit.Tuner("./ssd.yaml")
    ssd_tuner.tune(net, q_dataloader=val_data, eval_dataloader=val_dataset, eval_func=eval_func)
```

The iLiT tune() function will return a best quantized model under timeout constrain.