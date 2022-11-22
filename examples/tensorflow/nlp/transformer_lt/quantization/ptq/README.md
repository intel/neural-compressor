Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of Transformer-LT. This example can run on Intel CPUs and GPUs.

## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).

### 3. Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 4. Prepare Dataset & Pretrained model

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/transformer-lt-official-fp32-inference.tar.gz
tar -zxvf transformer-lt-official-fp32-inference.tar.gz
cd transformer-lt-official-fp32-inference
tar -zxvf transformer_lt_official_fp32_pretrained_model.tar.gz
```

Dataset is in data folder, pretrained model is in graph folder.

#### Automatic dataset & model download
Run the `prepare_dataset_model.sh` script located in `examples/tensorflow/nlp/transformer_lt/quantization/ptq`.

```shell
cd examples/tensorflow/nlp/transformer_lt/quantization/ptq
bash prepare_dataset_model.sh
```

## Run Command

```shell
python main.py --input_graph=/path/to/fp32_graphdef.pb --inputs_file=/path/to/newstest2014.en --reference_file=/path/to/newstest2014.de --vocab_file=/path/to/vocab.txt --config=./transformer_lt.yaml --tune
```

Details of enabling Intel® Neural Compressor on transformer-lt for Tensorflow.
=========================

This is a tutorial of how to enable transformer-lt model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For transformer-lt, we applied the latter one because we don't have dataset and metric for transformer-lt. The task is to implement the *q_dataloader* and *eval_func*.


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

### Write Yaml config file
In examples directory, there is a transformer_lt.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The transformer_lt_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

```yaml
model:
  name: transformer_lt
  framework: tensorflow
  inputs: input_tensor
  outputs: model/Transformer/strided_slice_19

device: cpu                                          # optional. default value is cpu, other value is gpu.

quantization:
  calibration:
    sampling_size: 500
  model_wise:
    weight:
      granularity: per_channel

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
    max_trials: 100
  random_seed: 9527
```

Here we set the input tensor and output tensors name into *inputs* and *outputs* field.
In this case we calibrate and quantize the model, and use our calibration dataloader initialized from a 'Dataset' object.

### Code update
After prepare step is done, we add tune code to generate quantized model.

```python
    from neural_compressor.experimental import Quantization
    from neural_compressor.adaptor.tf_utils.util import write_graph
    quantizer = Quantization(FLAGS.config)
    ds = Dataset(FLAGS.inputs_file, FLAGS.reference_file, FLAGS.vocab_file)
    quantizer.calib_dataloader = common.DataLoader(ds, collate_fn=collate_fn, batch_size=FLAGS.batch_size)
    quantizer.model = common.Model(graph)
    quantizer.eval_func = eval_func
    q_model = quantizer.fit()
    q_model.save(FLAGS.output_model)
```

The Intel® Neural Compressor quantizer.fit() function will return a best quantized model under time constraint.
