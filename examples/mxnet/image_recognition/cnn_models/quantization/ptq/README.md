Step-by-Step
============

This document is used to list steps of reproducing MXNet ResNet18_v1/ResNet50_v1/ResNet152_v1/Squeezenet1.0/MobileNet1.0/MobileNetv2_1.0/Inceptionv3 tuning zoo result.


# Prerequisite

### 1. Installation

  ```bash
  pip install -r requirements.txt
  ```
> Note: Supported MXNet [Version](../../../../../../README.md#supported-frameworks).

### 2. Prepare Dataset
  You can use `prepare_dataset.py` to download dataset for this example. like below:
  ```bash
  python prepare_dataset.py --dataset_location=./data
  ```
  This will download the val_256_q90.rec validation dataset to the **./data/** directory. 

### 2. Prepare Pre-trained model
  You can use `prepare_model.py` to download model for this example. like below:
  ```bash
  python prepare_model.py --model_name mobilenet1.0 --model_path ./model
  ```

  This will download the pre-trained **mobilenet1.0** model to the **./model/** directory.


# Run
### ResNet18_v1
```bash
bash run_tuning.sh --topology=resnet18_v1 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_resnet18
```

### ResNet50_v1
```bash
bash run_tuning.sh --topology=resnet50_v1 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_resnet50_v1
```
### ResNet152_v1
```bash
bash run_tuning.sh --topology=resnet152_v1 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_resnet152_v1
```
### SqueezeNet1
```bash
bash run_tuning.sh --topology=squeezenet1.0 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_squeezenet
```
### MobileNet1.0
```bash
bash run_tuning.sh --topology=mobilenet1.0 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_mobilenet1.0
```
### MobileNetv2_1.0
```bash
bash run_tuning.sh --topology=mobilenetv2_1.0 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_mobilenetv2_1.0
```
### Inception_v3
```bash
bash run_tuning.sh --topology=inceptionv3 --dataset_location=./data/val_256_q90.rec --input_model=/PATH/TO/MODEL --output_model=./nc_inception_v3
```

# Benchmark
Use resnet18 as an example:
```bash
# accuracy mode, run the whole test dataset and get accuracy
bash run_benchmark.sh --topology=resnet18_v1 --dataset_location=./data/val_256_q90.rec --input_model=./model --batch_size=32 --mode=accuracy
# benchmark mode, specify iteration number and batch_size in option, get throughput and latency
bash run_benchmark.sh --topology=resnet18_v1 --dataset_location=./data/val_256_q90.rec --input_model=./model --batch_size=32 --iters=100 --mode=benchmark
```

Examples of enabling Intel® Neural Compressor auto tuning on MXNet ResNet50
=======================================================

This is a tutorial of how to enable a MXNet classification model with Intel® Neural Compressor.

# User Code Analysis

Intel® Neural Compressor supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

>As ResNet50_v1/Squeezenet1.0/MobileNet1.0/MobileNetv2_1.0/Inceptionv3 series are typical classification models, use Top-K as metric which is built-in supported by Intel® Neural Compressor. So here we integrate MXNet ResNet with Intel® Neural Compressor by the first use case for simplicity.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandatory item for tuning. 


```
# conf.yaml

model:                                               # mandatory. used to specify model specific information.
  name: cnn
  framework: mxnet                                   # mandatory. possible values are tensorflow, mxnet, pytorch, pytorch_ipex, onnxrt_integerops and onnxrt_qlinearops.

evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.

```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.


### code update

After prepare step is done, we just need update imagenet_inference.py like below.

```python
    from neural_compressor.experimental import Quantization, common
    fp32_model = load_model(symbol_file, param_file, logger)
    quantizer = Quantization("./cnn.yaml")
    quantizer.model = common.Model(fp32_model)
    quantizer.calib_dataloader = data
    quantizer.eval_dataloader = data
    q_model = quantizer.fit()

```

The quantizer.fit() function will return a best quantized model during timeout constrain.
