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
cd examples\tensorflow\graph_networks\graphsage\quantization\ptq
pip install -r requirements.txt
```

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
Download Frozen graph:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/graphsage_frozen_model.pb
```

## 3. Prepare Dataset

```shell
wget https://snap.stanford.edu/graphsage/ppi.zip
unzip ppi.zip
```

# Run

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
  
  ```shell
  # The cmd of running faster_rcnn_resnet50
  bash run_quant.sh --input_model=./graphsage_frozen_model.pb --output_model=./nc_graphsage_int8_model.pb --dataset_location=./ppi
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./nc_graphsage_int8_model.pb  --dataset_location=./ppi --mode=performance
  ```

Details of enabling Intel® Neural Compressor on graphsage for Tensorflow.
=========================

This is a tutorial of how to enable graphsage model with Intel® Neural Compressor.
## User Code Analysis
User specifies fp32 *model*, calibration dataset *calib_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For graphsage, we applied the latter one because our philosophy is to enable the model with minimal changes. Hence we need to make two changes on the original code. The first one is to implement the q_dataloader and make necessary changes to *eval_func*.

### Code update

After prepare step is done, we just need update main.py like below.
```python
    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.data import DataLoader
        from neural_compressor.config import PostTrainingQuantConfig  
        dataset = CustomDataset()
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset, \
                                    batch_size=1, collate_fn = collate_function)          
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(args.input_graph, conf=conf, \
                                    calib_dataloader=calib_dataloader, eval_func=evaluate)
        q_model.save(args.output_graph)

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig()
            fit(args.input_graph, conf, b_func=evaluate)
        elif args.mode == 'accuracy':
            acc_result = evaluate(args.input_graph)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

```

The quantization.fit() function will return a best quantized model during timeout constrain.
