Step-by-Step
============

This document describes the step-by-step instructions for reproducing MXNet SSD-ResNet50_v1/SSD-Mobilenet 1.0 tuning results.



# Prerequisite
### 1. Installation

  ```Shell
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset

  If you want to use VOC2007 dataset, download [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) Raw image to the directory **~/.mxnet/datasets/voc** (Note:this path is unchangeable per original inference script requirement)

  If you want to use COCO2017 dataset, download [COCO2017](https://cocodataset.org/#download) Raw image to the directory **~/.mxnet/datasets/coco** (Note:this path is unchangeable per original inference script requirement)

  Or you can use `prepare_dataset.sh` to prepare dataset, like below:
  ```bash
  bash prepare_dataset.sh --data_path=./datasets --dataset=voc

  
  # help info
  bash prepare_dataset.sh -h

   Desc: Prepare dataset for MXNet Object Detection.

   -h --help              help info

   --dataset              set dataset category, voc or coco, default is voc.

   --data_path            directory of the download dataset, default is: /home/.mxnet/datasets/

  ```
# Run

### SSD-ResNet50_v1-VOC
```bash
bash run_tuning.sh --topology=ssd-resnet50_v1 --dataset_name=voc --dataset_location=/PATH/TO/DATASET --output_model=./lpot_ssd_resnet50_voc
```

### SSD-Mobilenet1.0-VOC
```bash
bash run_tuning.sh --topology=ssd-mobilenet1.0 --dataset_name=voc --dataset_location=/PATH/TO/DATASET --output_model=./lpot_ssd_mobilenet1.0_voc
```

### SSD-ResNet50_v1-COCO
```bash
bash run_tuning.sh --topology=ssd-resnet50_v1 --dataset_name=coco --dataset_location=/PATH/TO/DATASET --output_model=./lpot_ssd_resnet50_coco
```

### SSD-Mobilenet1.0-COCO
```bash
bash run_tuning.sh --topology=ssd-mobilenet1.0 --dataset_name=coco --dataset_location=/PATH/TO/DATASET --output_model=./lpot_ssd_mobilenet1.0_coco
```

# benchmark 
```bash
# accuracy mode, run the whole test dataset and get accuracy
bash run_benchmark.sh --topology=ssd-resnet50_v1 --dataset_name=voc --dataset_location=/PATH/TO/DATASET --input_model=/PATH/TO/MODEL_PREFIX --batch_size=32 --mode=accuracy 

# benchmark mode, specify iteration number and batch_size in option, get throughput and latency
bash run_benchmark.sh --topology=ssd-resnet50_v1 --dataset_name=voc --input_model=/PATH/TO/MODEL_PREFIX --batch_size=32 --iters=100 --mode=benchmark


```
For more detail, see:

```bash
  bash run_tuning.sh -h

   Desc: Run lpot MXNet Object Detection example.

   -h --help              help info

   --topology             model used for Object Detection, mobilenet1.0 or resnet50_v1, default is mobilenet1.0.

   --dataset_name         coco or voc, default is voc

   --dataset_location     location of dataset

   --input_model          prefix of fp32 model (eg: ./model/ssd-mobilenet )

   --output_model         Best tuning model by lpot will saved in this name prefix. default is './lpot_ssd_model'
```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on MXNet Object detection
=======================================================

This is a tutorial of how to enable a MXNet Object detection model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As this example use VOC/COCO dataset, use VOCMApMetrics/COCOEval as metric which is can find [here](https://github.com/dmlc/gluon-cv/blob/20a2ed3942720550728ce36c2be53b2d5bbbb6fd/gluoncv/utils/metrics/voc_detection.py#L13) and [here](https://cocodataset.org/). So we integrate MXNet SSD-ResNet50_v1/SSD-Mobilenet1.0 with Intel® Low Precision Optimization Tool by the second use case.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandatory items for tuning.


```
# conf.yaml

model:                                           
  name: ssd
  framework: mxnet

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

Because we use the second use case which need user to provide a custom "eval_func" which encapsulates the evaluation dataset and metric, we can not see a metric at config file tuning filed. We set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.


### code update

First, we need to construct evaluate function for Intel® Low Precision Optimization Tool. At eval_func, we get the val_dataset for the origin script, and return mAP metric to Intel® Low Precision Optimization Tool.

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

    # Doing auto-tuning here
    from lpot import Quantization
    quantizer = Quantization("./ssd.yaml")
    quantizer(net, q_dataloader=val_data, eval_dataloader=val_dataset, eval_func=eval_func)
```

The quantizer() function will return a best quantized model under timeout constrain.
