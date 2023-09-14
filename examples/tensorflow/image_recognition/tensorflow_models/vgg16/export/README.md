Step-by-Step
============

This document is used to show how to export Tensorflow FP32/INT8 QDQ model to ONNX FP32/INT8 QDQ model using Intel® Neural Compressor.


# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### Install requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this export ONNX INT8 model example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
Intel Extension for Tensorflow is mandatory to be installed for exporting Tensorflow model to ONNX.
```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

## 2. Prepare Model

The vgg16 checkpoint file comes from [models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
We can get the pb file by convert the checkpoint file.

  1. Download the checkpoint file from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  ```shell
  wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  tar -xvf vgg_16_2016_08_28.tar.gz
  ```

  2. Exporting the Inference Graph
  ```shell
  git clone https://github.com/tensorflow/models
  cd models/research/slim
  python export_inference_graph.py \
          --alsologtostderr \
          --model_name=vgg_16 \
          --output_file=/tmp/vgg_16_inf_graph.pb
  ```
  Make sure to use intel-tensorflow v1.15, and pip install tf_slim.
  #### Install Intel Tensorflow 1.15 up2
  Check your python version and use pip install 1.15.0 up2 from links below:
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp36-cp36m-manylinux2010_x86_64.whl                
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp35-cp35m-manylinux2010_x86_64.whl
  > Please note: The ImageNet dataset has 1001, the **VGG** and **ResNet V1** final layers have only 1000 outputs rather than 1001. So we need add the `--labels_offset=1` flag in the inference graph exporting command.

  3. Use [Netron](https://lutzroeder.github.io/netron/) to get the input/output layer name of inference graph pb, for vgg_16 the output layer name is `vgg_16/fc8/squeezed`

  4. Freezing the exported Graph, please use the tool `freeze_graph.py` in [tensorflow v1.15.2](https://github.com/tensorflow/tensorflow/blob/v1.15.2/tensorflow/python/tools/freeze_graph.py) repo 
  ```shell
  python freeze_graph.py \
          --input_graph=/tmp/vgg_16_inf_graph.pb \
          --input_checkpoint=./vgg_16.ckpt \
          --input_binary=true \
          --output_graph=./frozen_vgg16.pb \
          --output_node_names=vgg_16/fc8/squeezed
  ```

### 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

 ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/
  # convert validation subset
  bash prepare_imagenet_dataset.sh --output_dir=/path/to/imagenet/ --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_imagenet_dataset.sh --output_dir=/path/to/imagenet/ --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  cd vgg16/export
  ```
> **Note**: 
> The raw ImageNet data set resides in JPEG files should located in the following directory structure.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_dir/n01440764/ILSVRC2012_val_00000293.JPEG<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_dir/n01440764/ILSVRC2012_val_00000543.JPEG<br>
> where 'n01440764' is the unique synset label associated with these images.

## Run Command
Please note the dataset is TF records format for running quantization and benchmark.

### Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./frozen_vgg16.pb --output_model=./vgg_16.onnx --dtype=fp32 --quant_format=qdq
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./frozen_vgg16.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./frozen_vgg16.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./vgg_16.onnx --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./vgg_16.onnx --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./frozen_vgg16.pb --output_model=./frozen_vgg16_int8.onnx --dtype=int8 --quant_format=qdq --dataset_location=/path/to/imagenet/
```

## Run benchmark for Tensorflow INT8 model
```shell
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./frozen_vgg16_int8.onnx --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./frozen_vgg16_int8.onnx --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```