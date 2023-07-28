Step-by-Step
============

This document list steps of reproducing Vision Transformer model tuning results via Neural Compressor.

# Prerequisite

## 1. Environment

### Install Dependency Package

```
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

## 2. Prepare Pretrained model

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_11_0/HF-ViT-Base16-Img224-frozen.pb
```

## 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in ` examples/tensorflow/image_recognition/tensorflow_models/imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/
  # convert validation subset
  bash prepare_dataset.sh --output_dir=./vision_transformer/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_dataset.sh --output_dir=./vision_transformer/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  ```

# Run

## 1. Quantization

```
bash run_quant.sh --input_model <path to HF-ViT-Base16-Img224-frozen.pb> --output_model ./output --dataset_location <path to imagenet>
```


## 2. Benchmark

### Benchmark the fp32 model

```
bash run_benchmark.sh --input_model=<path to HF-ViT-Base16-Img224-frozen.pb> --mode=accuracy --dataset_location=<path to imagenet> --batch_size=32
```

### Benchmark the int8 model

```
bash run_benchmark.sh --input_model=./output.pb --mode=accuracy --dataset_location=<path to imagenet> --batch_size=32 --int8=true
```