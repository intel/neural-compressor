Step-by-Step
============

This document is used to show how to export Tensorflow Mobilenet_v2 FP32 model to ONNX FP32 model using Intel® Neural Compressor.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install requirements
```shell
pip install -r requirements.txt
```

### 3. Prepare Pretrained model

The mobilenet_v2 checkpoint file comes from [models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
We can get the pb file by convert the checkpoint file.

  1. Download the checkpoint file from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  ```shell
  wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
  tar -xvf mobilenet_v2_1.4_224.tgz
  ```

  2. Exporting the Inference Graph
  ```shell
  git clone https://github.com/tensorflow/models
  cd models/research/slim
  python export_inference_graph.py \
          --alsologtostderr \
          --model_name=mobilenet_v2 \
          --output_file=/tmp/mobilenet_v2_inf_graph.pb
  ```
  Make sure to use intel-tensorflow v1.15, and pip install tf_slim.
  #### Install Intel Tensorflow 1.15 up2
  Check your python version and use pip install 1.15.0 up2 from links below:
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp36-cp36m-manylinux2010_x86_64.whl                
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp35-cp35m-manylinux2010_x86_64.whl
  > Please note: The ImageNet dataset has 1001, the **VGG** and **ResNet V1** final layers have only 1000 outputs rather than 1001. So we need add the `--labels_offset=1` flag in the inference graph exporting command.
  3. Use [Netron](https://lutzroeder.github.io/netron/) to get the input/output layer name of inference graph pb, for vgg_16 the output layer name is `MobilenetV2/Predictions/Reshape_1`

  4. Freezing the exported Graph, please use the tool `freeze_graph.py` in [tensorflow v1.15.2](https://github.com/tensorflow/tensorflow/blob/v1.15.2/tensorflow/python/tools/freeze_graph.py) repo 
  ```shell
  python freeze_graph.py \
          --input_graph=/tmp/mobilenet_v2_inf_graph.pb \
          --input_checkpoint=./mobilenet_v2.ckpt \
          --input_binary=true \
          --output_graph=./frozen_mobilenet_v2.pb \
          --output_node_names=MobilenetV2/Predictions/Reshape_1
  ```


### 4. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/ImageNet. The dir include below folder and files:

```bash
ls /path/to/ImageNet
ILSVRC2012_img_val  val.txt
```

## Run Command

### Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./frozen_mobilenet_v2.pb --output_model=./mobilenet_v2.onnx
```

### Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./frozen_mobilenet_v2.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./frozen_mobilenet_v2.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```
Please note this dataset is TF records format.

### Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./mobilenet_v2.onnx --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
bash run_benchmark.sh --input_model=./mobilenet_v2.onnx --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
```
Please note this dataset is Raw image dataset.
