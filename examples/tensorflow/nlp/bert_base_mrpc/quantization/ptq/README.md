Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of bert base model on mrpc task.
This example can run on Intel CPUs and GPUs.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Tensorflow
```shell
pip install tensorflow
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### 3. Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest Intel GPU driver installation.
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

### 4. Prepare Dataset

#### Automatic dataset download
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert_base_mrpc/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_base_mrpc/quantization/ptq
python prepare_dataset.py --tasks='MRPC' --output_dir=./data
```

### 5. Prepare pretrained model

#### Automatic model download
Run the `prepare_model.sh` script located in `examples/tensorflow/nlp/bert_base_mrpc/quantization/ptq`.
NOTICE: This will need you first prepare your dataset as mrpc task need to train for good accuracy.


Usage:
```shell
cd examples/tensorflow/nlp/bert_base_mrpc/quantization/ptq
bash prepare_model.sh --dataset_location=./data --output_dir=./model
```

## Run Command
Make sure the data and model have been generated successfully which located at ./data and ./model respectively.
And your output_model will be located at ./output_model like the command below
  ```shell
    bash run_quant.sh --input_model=./model --dataset_location=./data --output_model=output_model
  ```
If you want the model without iterator inside the graph, you can add --strip_iterator like:
  ```shell
    bash run_quant.sh --input_model=./model --dataset_location=./data --output_model=output_model --strip_iterator
  ```
To run benchmark:
  ```shell
    bash run_benchmark.sh --input_model=./output_model.pb --init_checkpoint=./model --dataset_location=./data --batch_size=8 --mode=performance
  ```
  ```shell
        bash run_benchmark.sh --input_model=./output_model.pb --init_checkpoint=./model --dataset_location=./data --batch_size=8 --mode=accuracy
  ```

Details of enabling Intel® Neural Compressor on bert model for Tensorflow.
=========================

This is a tutorial of how to enable bert model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For bert, we applied the first one as we already have write dataset and metric for bert mrpc task. 

### Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    inputs=["input_file", "batch_size"],
    outputs=["loss/Softmax:0", "IteratorGetNext:3"],
    ...
    )
```
Here we set the input tensor and output tensors name into *inputs* and *outputs* args. In this case we calibrate and quantize the model, and use our calibration dataloader initialized from a 'Dataset' object.

### Code update

After prepare step is done, we add tune and benchmark code to generate quantized model and benchmark.

#### Tune
```python
    from neural_compressor import quantization
    from neural_compressor.config import PostTrainingQuantConfig
    from neural_compressor.data import DataLoader
    config = PostTrainingQuantConfig(
        inputs=["input_file", "batch_size"],
        outputs=["loss/Softmax:0", "IteratorGetNext:3"],
        calibration_sampling_size=[500],)
    calib_dataloader=DataLoader(dataset, collate_fn=collate_fn, framework='tensorflow')
    eval_dataloader=DataLoader(dataset, collate_fn=collate_fn, framework='tensorflow')
    q_model = quantization.fit(model=model, conf=config, calib_dataloader=calib_dataloader,
                    eval_dataloader=eval_dataloader, eval_metric=Accuracy())

    if FLAGS.strip_iterator:
        q_model.graph_def = strip_iterator(q_model.graph_def)
    q_model.save(FLAGS.output_model)
```
#### Benchmark
```python
    from neural_compressor.benchmark import fit
    from neural_compressor.config import BenchmarkConfig
    if FLAGS.mode == 'performance':
        conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
        fit(model, conf, b_func=evaluate)
    else:
        accuracy = evaluate(model.graph_def)
        print('Batch size = %d' % FLAGS.eval_batch_size)
        print("Accuracy: %.5f" % accuracy)

The Intel® Neural Compressor quantization.fit() function will return a best quantized model under time constraint.
