Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of DistilBERT base. This example can be run on Intel CPUs and GPUs.

## Model Details
This DistilBERT base model is based on the paper [*DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*](https://arxiv.org/abs/1910.01108). \
The [pretrained-model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you) thus used, was taken from [Hugging face model repository](https://huggingface.co/models). \
The frozen model pb can be found at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models/tree/master/models/language_modeling/tensorflow/distilbert_base/inference).

## Dataset Details
We use a part of Stanford Sentiment Treebank corpus for our task. Specifically, the validation split present in the SST2 dataset in the hugging face [repository](https://huggingface.co/datasets/sst2). It contains 872 labeled English sentences. The details for downloading the dataset are given below. 

## Prerequisite

### 1. Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### 2. Install TensorFlow 2.11.dev202242
Build a TensorFlow pip package from [intel-tensorflow spr_ww42 branch](https://github.com/Intel-tensorflow/tensorflow/tree/spr_ww42) and install it. How to build a TensorFlow pip package from source please refer to this [tutorial](https://www.tensorflow.org/install/source).

### 3. Install Requirements
```shell
pip install -r requirements.txt
```

### 4. Install Intel® Extension for TensorFlow

#### Quantizing the model on Intel GPU:
Intel® Extension for TensorFlow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU (Experimental):
Intel® Extension for TensorFlow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 5. Download Dataset
```shell
python download_dataset.py --path_to_save_dataset <enter path to save dataset>
```

## Run Command
### Run Tuning:
```shell
bash run_tuning.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --output_model=$OUTPUT_MODEL \
    --config=$CONFIG_FILE \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```
### Run Benchmark:
```shell
# benchmark mode: only get performance
bash run_benchmark.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --mode=benchmark \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --iters=$ITERS \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```

```shell
# accuracy mode: get performance and accuracy
bash run_benchmark.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --mode=accuracy \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```

Where (Default values are shown in the square brackets):
   * $INPUT_MODEL ["./distilbert_base_fp32.pb"]-- The path to input FP32 frozen model .pb file to load
   * $DATASET_DIR ["./sst2_validation_dataset"]-- The path to input dataset directory
   * $OUTPUT_MODEL ["./output_distilbert_base_int8.pb"]-- The user-specified export path to the output INT8 quantized model
   * $CONFIG_FILE ["./distilbert_base.yaml"]-- The path to quantization configuration .yaml file to load for tuning
   * $BATCH_SIZE [128]-- The batch size for model inference
   * $MAX_SEQ [128]-- The maximum total sequence length after tokenization
   * $ITERS [872]-- The number of iterations to run in benchmark mode, maximum value is 872
   * $WARMUPS [10]-- The number of warmup steps before benchmarking the model, maximum value is 22
   * $INTER_THREADS [2]-- The number of inter op parallelism thread to use, which can be set to the number of sockets
   * $INTRA_THREADS [28]-- The number of intra op parallelism thread to use, which can be set to the number of physical core per socket


Details of enabling Intel® Neural Compressor on DistilBERT base for TensorFlow
=========================

This is a tutorial of how to enable DistilBERT base model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataloader *q_dataloader*, evaluation dataloader *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataloader *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataloader and metric by itself.

For DistilBERT base, we applied the latter one. The task is to implement the *q_dataloader* and *eval_func*.


### q_dataloader Part Adaption
Below dataloader class uses generator function to provide the model with input.

```python
class Dataloader(object):
    def __init__(self, data_location, batch_size, steps):
        self.batch_size = batch_size
        self.data_location = data_location
        self.num_batch = math.ceil(steps / batch_size)

    def __iter__(self):
        return self.generate_dataloader(self.data_location).__iter__()

    def __len__(self):
        return self.num_batch

    def generate_dataloader(self, data_location):
        dataset = load_dataset(data_location)
        for batch_id in range(self.num_batch):
            feed_dict, labels = create_feed_dict_and_labels(dataset, batch_id, self.num_batch)
            yield feed_dict, labels
```

### Write Yaml Config File
In examples directory, there is a distilbert_base.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The distilbert_base_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

```yaml
model:
  name: distilbert_base
  framework: tensorflow

device: cpu                # optional. default value is cpu, other value is gpu.

quantization:
  calibration:
    sampling_size: 500
  model_wise:
    weight:
      granularity: per_channel

tuning:
  accuracy_criterion:
    relative: 0.02
  exit_policy:
    timeout: 0
    max_trials: 100
    performance_only: False
  random_seed: 9527
```

In this case we calibrate and quantize the model, and use our user-defined calibration dataloader.

### Code Update
After prepare step is done, we add the code for quantization tuning to generate quantized model.

```python
from neural_compressor.experimental import Quantization, common
quantizer = Quantization(ARGS.config)
quantizer.calib_dataloader = self.dataloader
quantizer.model = common.Model(graph)
quantizer.eval_func = self.eval_func 
q_model = quantizer.fit()
```

The Intel® Neural Compressor quantizer.fit() function will return a best quantized model under time constraint.
