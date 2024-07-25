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
cd examples/3.x_api/tensorflow/object_detection
pip install -r requirements.txt
cd mask_rcnn_inception_v2/quantization/ptq
```

### Install Protocol Buffer Compiler

`Protocol Buffer Compiler` in version higher than 3.0.0 is necessary ingredient for automatic COCO dataset preparation. To install please follow
[Protobuf installation instructions](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager).

### Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[xpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare Model

```shell
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

## 3. Prepare Dataset

### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/3.x_api/tensorflow/object_detection`.

Usage:
```shell
cd examples/3.x_api/tensorflow/object_detection/
. prepare_dataset.sh
cd mask_rcnn_inception_v2/quantization/ptq
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).


# Run

Now we support both pb and ckpt formats.

## 1. Quantization
### For PB format
  
  ```shell
  bash run_quant.sh --input_model=./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --output_model=./tensorflow-mask_rcnn_inception_v2-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

### For ckpt format
  
  ```shell
  bash run_quant.sh --input_model=./mask_rcnn_inception_v2_coco_2018_01_28/ --output_model=./tensorflow-mask_rcnn_inception_v2-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

## 2. Benchmark
  ```shell
  # run performance benchmark
  bash run_benchmark.sh --input_model=./tensorflow-mask_rcnn_inception_v2-tune.pb  --dataset_location=/path/to/dataset/coco_val.record --mode=performance

  # run accuracy benchmark
  bash run_benchmark.sh --input_model=./tensorflow-mask_rcnn_inception_v2-tune.pb  --dataset_location=/path/to/dataset/coco_val.record --mode=accuracy
  ```

Details of enabling Intel® Neural Compressor on mask_rcnn_inception_v2 for Tensorflow.
=========================

This is a tutorial of how to enable mask_rcnn_inception_v2 model with Intel® Neural Compressor.
## User Code Analysis
User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For mask_rcnn_inception_v2, we applied the latter one because our philosophy is to enable the model with minimal changes. Hence we need to make two changes on the original code. The first one is to implement the q_dataloader and make necessary changes to *eval_func*.

### Code update

After prepare step is done, we just need update main.py like below.
```python
    if args.tune:
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model, Model

        quant_config = StaticQuantConfig(weight_granularity="per_channel")
        model = Model(args.input_graph)
        model.input_tensor_names = ['image_tensor']
        model.output_tensor_names = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
        q_model = quantize_model(model, quant_config, calib_dataloader)
        q_model.save(args.output_model)
            
    if args.benchmark:
        if args.mode == 'performance':
            evaluate(args.input_graph)
        else:
            accuracy = evaluate(args.input_graph)
            print('Batch size = %d' % args.batch_size)
            print("Accuracy: %.5f" % accuracy)
```

The quantization.fit() function will return a best quantized model during timeout constrain.
