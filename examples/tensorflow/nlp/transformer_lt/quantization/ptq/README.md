Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of Transformer-LT. This example can run on Intel CPUs and GPUs.

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
pip install --upgrade intel-extension-for-tensorflow[xpu]
```
Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest Intel GPU driver installation.
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

### 4. Prepare Dataset & Pretrained model

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/transformer-lt-official-fp32-inference.tar.gz
tar -zxvf transformer-lt-official-fp32-inference.tar.gz
cd transformer-lt-official-fp32-inference
tar -zxvf transformer_lt_official_fp32_pretrained_model.tar.gz
```

Dataset is in data folder, pretrained model is in graph folder.

#### Automatic dataset & model download
Run the `prepare_dataset_model.sh` script located in `examples/3.x_api/tensorflow/nlp/transformer_lt/quantization/ptq`.

```shell
cd examples/3.x_api/tensorflow/nlp/transformer_lt/quantization/ptq
bash prepare_dataset_model.sh
```

## Run Command
### Quantization

```shell
bash run_quant.sh --input_model=./model/fp32_graphdef.pb --dataset_location=./data --output_model=./model/int8_graphdef.pb
```
### Benchmark
```shell
bash run_benchmark.sh --input_model=./model/int8_graphdef.pb --dataset_location=./data --mode=performance

bash run_benchmark.sh --input_model=./model/int8_graphdef.pb --dataset_location=./data --mode=accuracy --batch_size=1
```

Details of enabling Intel® Neural Compressor on transformer-lt for Tensorflow.
=========================

This is a tutorial of how to enable transformer-lt model with Intel® Neural Compressor.

### q_dataloader Part Adaption
Below dataset class uses getitem to provide the model with input.

```python
class Dataset(object):
    def __init__(self, *args):
        # initialize dataset related info here
        ...

    def __getitem__(self, index):
        data = self.batch[index]
        label = self.ref_lines[index]
        return data[0], label

    def __len__(self):
        return len(self.batch)
```

### Evaluation Part Adaption
We evaluate the model with BLEU score, its source: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py

Here we set the input tensor and output tensors name into *inputs* and *outputs* args.
In this case we calibrate and quantize the model, and use our calibration dataloader initialized from a 'Dataset' object.

### Code update
After prepare step is done, we add tune code to generate quantized model.

#### Tune
```python
    from neural_compressor.tensorflow import StaticQuantConfig, quantize_model, Model

    dataset = Dataset(FLAGS.inputs_file, FLAGS.reference_file, FLAGS.vocab_file)
    calib_dataloader = BaseDataLoader(dataset=dataset, batch_size=FLAGS.batch_size, collate_fn=collate_fn)

    quant_config = StaticQuantConfig()
    model = Model(graph)
    model.input_tensor_names = ['input_tensor']
    model.output_tensor_names = ['model/Transformer/strided_slice_19']
    q_model = quantize_model(model, quant_config, calib_dataloader)
    try:
        q_model.save(FLAGS.output_model)
    except Exception as e:
        print("Failed to save model due to {}".format(str(e)))
```
#### Benchmark
```python
    if FLAGS.benchmark:
        assert FLAGS.mode == 'performance' or FLAGS.mode == 'accuracy', \
        "Benchmark only supports performance or accuracy mode."
        acc = eval_func(graph)
        if FLAGS.mode == 'accuracy':
            print('Accuracy is {:.3f}'.format(acc))
```
