Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of Transformer_LT_mlperf. Part of the inference code is based on the transformer mlperf evaluation code. Detailed information on mlperf benchmark can be found in [mlcommons/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer).

## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### 2. Install TensorFlow
```shell
pip install tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#validated-software-environment).

### 3. Prepare Dataset & Frozen Model
Follow the [instructions](https://github.com/IntelAI/models/blob/master/benchmarks/language_translation/tensorflow/transformer_mlperf/inference/fp32/README.md) on [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models) to download and preprocess the WMT English-German dataset and generate a FP32 frozen model. Please make sure there are the following files in the dataset directory and the input model directory.
* DATASET_DIR: newstest2014.de, newstest2014.en, vocab.ende.32768

* INPUT_MODEL_DIR: the FP32 frozen model .pb file

## Run Command
### Run Tuning:
```
bash run_tuning.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --output_model=$OUTPUT_MODEL \
    --file_out=$OUTPUT_TRANSLATION_FILE \
    --config=$CONFIG_FILE \
    --batch_size=$BATCH_SIZE \
    --warmup_steps=$WARMUPS \
    --bleu_variant=$VARIANT \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```
### Run Benchmark:
```
# benchmark mode: only get performance
bash run_benchmark.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --file_out=$OUTPUT_TRANSLATION_FILE \
    --mode=benchmark \
    --batch_size=$BATCH_SIZE \
    --iters=$ITERATIONS \
    --warmup_steps=$WARMUPS \
    --bleu_variant=$VARIANT \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```

```
# accuracy mode: get performance and accuracy
bash run_benchmark.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --file_out=$OUTPUT_TRANSLATION_FILE \
    --mode=accuracy \
    --batch_size=$BATCH_SIZE \
    --warmup_steps=$WARMUPS \
    --bleu_variant=$VARIANT \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```

Where (Default values are shown in the square brackets):
   * $INPUT_MODEL ["./transformer_mlperf_fp32.pb"]-- The path to input FP32 frozen model .pb file to load
   * $DATASET_DIR ["./transformer_uniform_data"]-- The path to input dataset directory, which should include newstest2014.en, newstest2014.de and vocab.ende.32768
   * $OUTPUT_MODEL ["./output_transformer_mlperf_int8.pb"]-- The user-specified export path to the output INT8 quantized model
   * $OUTPUT_TRANSLATION_FILE ["./output_translation_result.txt"] -- The file path to save model translation result, only the most recent translation result will be saved
   * $CONFIG_FILE ["./transformer_lt_mlperf.yaml"]-- The path to quantization configuration .yaml file to load for tuning
   * $BATCH_SIZE [64]-- The batch size for model inference
   * $ITERATIONS [-1]-- The user-defined total inference iterations in benchmark mode. If it is -1, it means to run the entire dataset
   * $WARMUPS [5]-- The number of warmup steps before benchmarking the model
   * $VARIANT ["uncased"]-- The case sensitive type to compute BLEU score, which is one of two options: 'uncased'/'cased'
   * $INTER_THREADS [2]-- The number of inter op parallelism thread to use, which can be set to the number of sockets
   * $INTRA_THREADS [56]-- The number of intra op parallelism thread to use, which can be set to the number of cores


Details of enabling Intel® Neural Compressor on Transformer_LT_mlperf for Tensorflow.
=========================

This is a tutorial of how to enable Transformer_LT_mlperf model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For Transformer_LT_mlperf, we applied the latter one because we don't have dataset and metric for Transformer_LT_mlperf. The task is to implement the *q_dataloader* and *eval_func*.


### q_dataloader Part Adaption
Below dataset class uses getitem to provide the model with input.

```python
class Dataset(object):
    def __init__(self, *args):
        # initialize dataset related info here
        ...

    def __getitem__(self, index):
        return self.lines[index], 0

    def __len__(self):
        return len(self.lines)
```

### Evaluation Part Adaption
We evaluate the model with BLEU score, its source: https://github.com/IntelAI/models/blob/master/models/language_translation/tensorflow/transformer_mlperf/inference/fp32/transformer/compute_bleu.py

### Write Yaml config file
In examples directory, there is a transformer_lt_mlperf.yaml. We could remove most of items and only keep mandatory item for tuning.

```yaml
model:
  name: transformer_lt_mlperf
  framework: inteltensorflow
  inputs: input_tokens
  outputs: model/Transformer/strided_slice_15

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
from neural_compressor.experimental import Quantization, common
quantizer = Quantization(FLAGS.config)
dataset = Dataset(FLAGS.input_file, FLAGS.vocab_file)
quantizer.calib_dataloader = common.DataLoader(dataset,
                                               collate_fn = collate_fn,
                                               batch_size = FLAGS.batch_size)
quantizer.model = common.Model(graph)
quantizer.eval_func = eval_func
q_model = quantizer.fit()
q_model.save(FLAGS.output_model)
```

The Intel® Neural Compressor quantizer.fit() function will return a best quantized model under time constraint.
