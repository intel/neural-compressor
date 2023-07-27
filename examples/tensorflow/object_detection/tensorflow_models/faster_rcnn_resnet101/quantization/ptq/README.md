Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Object Detection models tuning results. This example can run on Intel CPUs and GPUs.

# Prerequisite


## 1. Environment
Recommend python 3.6 or higher version.

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/tensorflow_models/
pip install -r requirements.txt
cd faster_rcnn_resnet101/quantization/ptq
```

### Install Protocol Buffer Compiler

`Protocol Buffer Compiler` in version higher than 3.0.0 is necessary ingredient for automatic COCO dataset preparation. To install please follow
[Protobuf installation instructions](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager).

### Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare Model

```shell
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

## 3. Prepare Dataset

### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/tensorflow_models/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/object_detection/tensorflow_models/
. prepare_dataset.sh
cd faster_rcnn_resnet101/quantization/ptq
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).


# Run

Now we support both pb and saved_model formats.

## Quantization Config

The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## 1. Quantization
### For PB format
  
  ```shell
  # The cmd of running faster_rcnn_resnet101
  bash run_quant.sh --input_model=./faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb --output_model=./tensorflow-faster_rcnn_resnet101-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

### For saved_model format
  
  ```shell
  # The cmd of running faster_rcnn_resnet101
  bash run_quant.sh --input_model=./faster_rcnn_resnet101_coco_2018_01_28/saved_model/ --output_model=./tensorflow-faster_rcnn_resnet101-tune --dataset_location=/path/to/dataset/coco_val.record
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./tensorflow-faster_rcnn_resnet101-tune.pb  --dataset_location=/path/to/dataset/coco_val.record --mode=performance
  ```

Details of enabling Intel® Neural Compressor on faster_rcnn_resnet101 for Tensorflow.
=========================

This is a tutorial of how to enable faster_rcnn_resnet101 model with Intel® Neural Compressor.
## User Code Analysis
User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For faster_rcnn_resnet101, we applied the latter one because our philosophy is to enable the model with minimal changes. Hence we need to make two changes on the original code. The first one is to implement the q_dataloader and make necessary changes to *eval_func*.

### Code update

After prepare step is done, we just need update main.py like below.
```python
    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
        accuracy_criterion = AccuracyCriterion(criterion='absolute')
        config = PostTrainingQuantConfig(calibration_sampling_size=[10, 50, 100, 200],
                                         excluded_precisions=["bf16"],
                                         accuracy_criterion=accuracy_criterion)
        q_model = quantization.fit(model=args.input_graph, conf=config, 
                                    calib_dataloader=calib_dataloader, eval_func=evaluate)
        q_model.save(args.output_model)
            
    if args.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if args.mode == 'performance':
            conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
            fit(args.input_graph, conf, b_func=evaluate)
        else:
            accuracy = evaluate(args.input_graph)
            print('Batch size = %d' % args.batch_size)
            print("Accuracy: %.5f" % accuracy)
```

The quantization.fit() function will return a best quantized model during timeout constrain.
