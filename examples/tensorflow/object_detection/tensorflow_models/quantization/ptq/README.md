Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Object Detection models tuning results.
Currently, we've enabled below models.
 * ssd_resnet50_v1
 * ssd_resnet34
 * ssd_mobilenet_v1
 * fastrcnn_inception_resnet_v2
 * fastrcnn_resnet101
 * fastrcnn_resnet50
 * maskrcnn_inception_v2
## Prerequisite


### 1. Installation
Recommend python 3.6 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).
### 3. Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/tensorflow_models/quantization/ptq
pip install -r requirements.txt
```

### 4. Install Protocol Buffer Compiler

`Protocol Buffer Compiler` in version higher than 3.0.0 is necessary ingredient for automatic COCO dataset preparation. To install please follow
[Protobuf installation instructions](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager).

### 5. Prepare Dataset

#### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/tensorflow_models/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/object_detection/tensorflow_models/quantization/ptq
. prepare_dataset.sh
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

#### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

### 6. Download Model

#### Automated approach
Run the `prepare_model.py` script located in `examples/tensorflow/object_detection/tensorflow_models/quantization/ptq`.

```
usage: prepare_model.py [-h] [--model_name {ssd_resnet50_v1,ssd_mobilenet_v1}]
                        [--model_path MODEL_PATH]

Prepare pre-trained model for COCO object detection

optional arguments:
  -h, --help            show this help message and exit
  --model_name {ssd_resnet50_v1,ssd_mobilenet_v1}
                        model to download, default is ssd_resnet50_v1
  --model_path MODEL_PATH
                        directory to put models, default is ./model
```

#### Manual approach

##### ssd_resnet50_v1
```shell
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz -C /tmp
```

##### ssd_mobilenet_V1

```shell
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

##### faster_rcnn_inception_resnet_v2

```shell
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

##### faster_rcnn_resnet101

```shell
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

##### faster_rcnn_resnet50

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
tar -xvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
```

##### mask_rcnn_inception_v2

```shell
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

##### ssd_resnet34
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
```
You need to install intel-tensorflow==2.4.0 to enable ssd_resnet34 model.

## Run Command

Now we support both pb and ckpt formats.

### For PB model
  
  ```shell
  # The cmd of running ssd_resnet50_v1
  bash run_tuning.sh --config=ssd_resnet50_v1.yaml --input_model=/tmp/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb --output_model=./tensorflow-ssd_resnet50_v1-tune.pb
  ```

### For ckpt model
  
  ```shell
  # The cmd of running ssd_resnet50_v1
  bash run_tuning.sh --config=ssd_resnet50_v1.yaml --input_model=/tmp/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ --output_model=./tensorflow-ssd_resnet50_v1-tune.pb
  ```
> Note
>
> 1. Make sure to add dataset_location=/path/to/dataset/coco_val.record in config file: "ssd_resnet50_v1.yaml"
>
> 2. For ssd_resnet34 model, anno_path of evaluation/accuracy/metric/COCOmAP in config file should be "label_map.yaml"

Details of enabling Intel® Neural Compressor on ssd_resnet50_v1 for Tensorflow.
=========================

This is a tutorial of how to enable ssd_resnet50_v1 model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For ssd_resnet50_v1, we applied the latter one because our philosophy is to enable the model with minimal changes. Hence we need to make two changes on the original code. The first one is to implement the q_dataloader and make necessary changes to *eval_func*.


### q_dataloader Part Adaption
Specifically, we need to add one generator to iterate the dataset per Intel® Neural Compressor requirements. The easiest way is to implement *__iter__* interface. Below function will yield the images to feed the model as input.

```python
def __iter__(self):
    """Enable the generator for q_dataloader

    Yields:
        [Tensor]: images
    """
    data_graph = tf.Graph()
    with data_graph.as_default():
        self.input_images, self.bbox, self.label, self.image_id = self.get_input(
        )

    self.data_sess = tf.compat.v1.Session(graph=data_graph,
                                          config=self.config)
    for i in range(COCO_NUM_VAL_IMAGES):
        input_images = self.data_sess.run([self.input_images])
        yield input_images
```

### Evaluation Part Adaption
The Class model_infer has the run_accuracy function which actually could be re-used as the eval_func.

Compare with the original version, we added the additional parameter **input_graph** as the Intel® Neural Compressor would call this interface with the graph to be evaluated. The following code snippet also need to be added into the run_accuracy function to update the class members like self.input_tensor and self.output_tensors.
```python
if input_graph:
    graph_def = get_graph_def(self.args.input_graph, self.output_layers)
    input_graph = tf.Graph()
    with input_graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    self.infer_graph = input_graph
    # Need to reset the input_tensor/output_tensor
    self.input_tensor = self.infer_graph.get_tensor_by_name(
        self.input_layer + ":0")
    self.output_tensors = [
        self.infer_graph.get_tensor_by_name(x + ":0")
        for x in self.output_layers
    ]
```

### Write Yaml config file
In examples directory, there is a ssd_resnet50_v1.yaml. We could remove most of items and only keep mandatory item for tuning.

```yaml
model:                                               # mandatory. used to specify model specific information.
  name: ssd_resnet50_v1
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.
  inputs: image_tensor
  outputs: num_detections,detection_boxes,detection_scores,detection_classes

quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 100                               # optional. default value is 100. used to set how many samples should be used in calibration.
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
    activation:
      algorithm: minmax
    weight:
      algorithm: minmax
  op_wise: {
             'FeatureExtractor/resnet_v1_50/fpn/bottom_up_block5/Conv2D': {
               'activation':  {'dtype': ['fp32']},
             },
             'WeightSharedConvolutionalBoxPredictor_2/ClassPredictionTower/conv2d_0/Conv2D': {
               'activation':  {'dtype': ['fp32']},
             }
           }

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
    max_trials: 100                                  # optional. max tune times. default value is 100. combine with timeout field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.
```
Here we set the input tensor and output tensors name into *inputs* and *outputs* field. Meanwhile, we set mAp target as tolerating 0.01 relative mAp of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

### Code update

After prepare step is done, we just need update infer_detections.py like below.
```python
from neural_compressor.experimental import Quantization,common

quantizer = Quantization(args.config)
quantizer.model = common.Model(args.input_graph)
quantizer.calib_dataloader = infer
quantizer.eval_dataloader = infer
quantizer.eval_func = infer.accuracy_check
q_model = quantizer.fit()
```

The quantizer.fit() function will return a best quantized model during timeout constrain.
