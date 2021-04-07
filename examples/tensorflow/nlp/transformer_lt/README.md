Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Low Precision Optimization Tool tuning zoo result of Transformer-LT.

## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```

### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../README.md).

### 3. Prepare Dataset & Pretrained model

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/transformer-lt-official-fp32-inference.tar.gz
tar -zxvf transformer-lt-official-fp32-inference.tar.gz
cd transformer-lt-official-fp32-inference
tar -zxvf transformer_lt_official_fp32_pretrained_model.tar.gz
```

Dataset is in data folder, pretrained model is in graph folder.

#### Automatic dataset & model download
Run the `prepare_dataset_model.sh` script located in `examples/tensorflow/nlp/transformer_lt`.

```shell
cd examples/tensorflow/nlp/transformer_lt
bash prepare_dataset_model.sh
```

## Run Command

```shell
python main.py --input_graph=/path/to/fp32_graphdef.pb --inputs_file=/path/to/newstest2014.en --reference_file=/path/to/newstest2014.de --vocab_file=/path/tp/vocab.txt --config=./transformer_lt.yaml --tune
```

Details of enabling Intel® Low Precision Optimization Tool on transformer-lt for Tensorflow.
=========================

This is a tutorial of how to enable transformer-lt model with Intel® Low Precision Optimization Tool.
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
In examples directory, there is a transformer_lt.yaml. We could remove most of items and only keep mandatory item for tuning.

```yaml
model:
  name: transformer_lt
  framework: tensorflow
  inputs: input_tensor
  outputs: model/Transformer/strided_slice_19

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
    from lpot.experimental import Quantization
    from lpot.adaptor.tf_utils.util import write_graph
    quantizer = Quantization(FLAGS.config)
    ds = Dataset(FLAGS.inputs_file, FLAGS.reference_file, FLAGS.vocab_file)
    quantizer.calib_dataloader = common.DataLoader(ds, collate_fn=collate_fn, batch_size=FLAGS.batch_size)
    quantizer.model = common.Model(graph)
    quantizer.eval_func = eval_func
    q_model = quantizer()
    q_model.save(FLAGS.output_model)
```

The Intel® Low Precision Optimization Tool quantizer() function will return a best quantized model under time constraint.
