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

### 5. Download Dataset
```shell
python download_dataset.py --path_to_save_dataset <enter path to save dataset>
```

### 6. Download Model
Download Frozen graph:
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/distilbert_frozen_graph_fp32_final.pb
```

## Run Command
### Run Tuning:
```shell
bash run_quant.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --output_model=$OUTPUT_MODEL \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```
### Run Benchmark:
```shell
# performance mode: get performance
bash run_benchmark.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --mode=performance \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --iters=$ITERS \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS
```

```shell
# accuracy mode: get accuracy
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
   * $BATCH_SIZE [128]-- The batch size for model inference
   * $MAX_SEQ [128]-- The maximum total sequence length after tokenization
   * $ITERS [872]-- The number of iterations to run in benchmark mode, maximum value is 872
   * $WARMUPS [10]-- The number of warmup steps before benchmarking the model, maximum value is 22
   * $INTER_THREADS [2]-- The number of inter op parallelism thread to use, which can be set to the number of sockets
   * $INTRA_THREADS [28]-- The number of intra op parallelism thread to use, which can be set to the number of physical core per socket


### Run Smooth Quant to improve int8 accuracy

#### Tuning
```shell
bash run_quant.sh \
    --input_model=$INPUT_MODEL \
    --dataset_location=$DATASET_DIR \
    --output_model=$OUTPUT_MODEL \
    --batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ \
    --warmup_steps=$WARMUPS \
    --num_inter=$INTER_THREADS \
    --num_intra=$INTRA_THREADS \
    --sq=True
```


Details of enabling Intel® Neural Compressor on DistilBERT base for TensorFlow
=========================

This is a tutorial of how to enable DistilBERT base model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataloader *q_dataloader*, evaluation dataloader *eval_dataloader* and metric.

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

### Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

### Code Update
After prepare step is done, we add the code for quantization tuning to generate quantized model.

#### Tune
```python
    from neural_compressor import quantization
    from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.02)
    config = PostTrainingQuantConfig(calibration_sampling_size=[500],
                                        accuracy_criterion=accuracy_criterion)
    q_model = quantization.fit(model=graph, conf=config, calib_dataloader=self.dataloader,
                    eval_func=self.eval_func)
    try:
        q_model.save(ARGS.output_graph)
    except Exception as e:
        tf.compat.v1.logging.error("Failed to save model due to {}".format(str(e)))
```
#### Benchmark
```python
    from neural_compressor.benchmark import fit
    from neural_compressor.config import BenchmarkConfig
    if ARGS.mode == 'performance':
        conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
        fit(graph, conf, b_func=self.eval_func)
    elif ARGS.mode == 'accuracy':
        self.eval_func(graph)
```

The Intel® Neural Compressor quantization.fit() function will return a best quantized model under time constraint.
