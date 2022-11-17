Step-by-Step
============

This document list steps of reproducing Intel Optimized TensorFlow image recognition models tuning results via Neural Compressor.
This example can run on Intel CPUs and GPUs.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x.
> [Version support](../../../../../../README.md#supported-frameworks)
# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```shell
  cd examples/tensorflow/semantic_image_segmentation/deeplab/quantization/ptq
  pip install -r requirements.txt
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

### 4. Prepare Dataset
Please use the script under the folder `datasets` to download and convert PASCAL VOC 2012 semantic segmentation dataset to TFRecord. Refer to [Running DeepLab on PASCAL VOC 2012 Semantic Segmentation Dataset](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/pascal.md#running-deeplab-on-pascal-voc-2012-semantic-segmentation-dataset) for more details.
```shell
# From the examples/tensorflow/semantic_image_segmentation/deeplab/datasets directory.
sh download_and_convert_voc2012.sh
```

### 5. Prepare pre-trained model
Refer to [Export trained deeplab model to frozen inference graph](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/export_model.md#export-trained-deeplab-model-to-frozen-inference-graph) for more details.

1. Download the checkpoint file
```shell
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
tar -xvf deeplabv3_pascal_train_aug_2018_01_04.tar.gz
```
2. Export to a frozen graph
```shell
git clone https://github.com/tensorflow/models.git
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python deeplab/export_model.py \ 
    --checkpoint_path=/PATH/TO/deeplabv3_pascal_train_aug/model.ckpt \ 
    --export_path=/PATH/TO/deeplab_export.pb \ 
    --model_variant="xception_65"  \ 
    --logtostderr \ 
    --eval_split="val" \ 
    --model_variant="xception_65" \ 
    --atrous_rates=6 \ 
    --atrous_rates=12 \ 
    --atrous_rates=18 \ 
    --output_stride=16 \ 
    --decoder_output_stride=4 \ 
    --eval_crop_size="513,513" \ 
    --dataset="pascal_voc_seg"
```

# Run
```shell
cd examples/tensorflow/semantic_image_segmentation/deeplab/quantization/ptq
bash run_tuning.sh --config=deeplab.yaml --input_model=/PATH/TO/deeplab_export.pb --output_model=./nc_deeplab.pb
```


Examples of enabling Intel® Neural Compressor auto tuning on Deeplab model for tensorflow
=======================================================

This is a tutorial of how to enable deeplab model with Intel® Neural Compressor.

# User Code Analysis

Intel® Neural Compressor supports two usages:

1. User specifies fp32 "model", yaml configured calibration dataloader in calibration field and evaluation dataloader in evaluation field, metric in tuning.metric field of model-specific yaml config file.

> *Note*: 
> you should change the model-specific yaml file dataset path to your own dataset path

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

We provide Deeplab model pretrained on PASCAL VOC 2012, Using mIOU as metric which is built-in supported by Intel® Neural Compressor.

### Write Yaml config file

In examples directory, there is a deeplab.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The deeplab_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

```yaml
# deeplab.yaml
device: cpu                                          # optional. default value is cpu, other value is gpu.
model:                                               # mandatory. neural_compressor uses this model name and framework name to decide where to save tuning history and deploy yaml.
  name: deeplab
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.

quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 50, 100                           # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      batch_size: 1
      dataset:
        VOCRecord:
          root: /path/to/pascal_voc_seg/tfrecord     # NOTE: modify to calibration 
      transform:
        ParseDecodeVoc: {}


evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      mIOU: 
        num_classes: 21                              # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 1
      dataset:
        VOCRecord:
          root: /path/to/pascal_voc_seg/tfrecord     # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeVoc: {}
  performance:                                       # optional. used to benchmark performance of passing model.
    iteration: 100
    configs:
      cores_per_instance: 4
      num_of_instance: 6
    dataloader:
      batch_size: 1
      dataset:
        VOCRecord:
          root: /path/to/pascal_voc_seg/tfrecord     # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeVoc: {}

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.


```

Here we choose topk which is built-in metric and set accuracy criterion as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as long as a tuning config meet accuracy target.

### Tune

After completed preparation steps, we just need to add below tuning part in `eval_classifier_optimized_graph` class.

```python
from neural_compressor.experimental import Quantization, common
quantizer = Quantization(self.args.config)
quantizer.model = common.Model(self.args.input_graph)
q_model = quantizer.fit()
q_model.save(self.args.output_graph)
```

### Benchmark
```python
from neural_compressor.experimental import Benchmark, common
evaluator = Benchmark(self.args.config)
evaluator.model = common.Model(self.args.input_graph)
evaluator(self.args.mode)
```

