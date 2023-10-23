Step-by-Step
============

This example load [BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)(BEiT) model and confirm its accuracy and performance based on [ImageNet-1k dataset](http://www.image-net.org/). You need to download this dataset yourself.

In this example, the BEiT model is pre-trained in a self-supervised fashion on ImageNet-22k - also called ImageNet-21k (14 million images, 21,841 classes) at resolution 224x224, and fine-tuned on the same dataset at resolution 224x224. It was first released in [this repository](https://github.com/microsoft/unilm/tree/master/beit). 


# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Prepare DETR R18 model for table structure recognition.

```shell
python prepare_model.py --input_model=beit_base_patch16_224 --output_model=beit_base_patch16_224_pt22k_ft22kto1k.onnx
```

## 3. Prepare Dataset

Download and extract [ImageNet-1k](http://www.image-net.org/) to dir: /path/to/imagenet. The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

## 1. Quantization

Quantize model with QLinearOps:

```bash
bash run_quant.sh --input_model=/path/to/model \  # model path as *.onnx
                   --dataset_location=/path/to/imagenet \
                   --output_model=/path/to/save \
                   --quant_format="QOperator"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/imagenet \
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```