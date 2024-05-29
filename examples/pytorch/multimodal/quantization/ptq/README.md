# Run Quantization on Multimodal Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as LLaVA. 

## Install
If you are not using Linux, do NOT proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process. To align the official implementation of [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning)

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire. 

## Run quantization

