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

## Download the evaluation datasets for visual question-answering (VQA) tasks
We use TextVQA as an evaluation benchmark in our example. Please following instructions below to prepare evaluation data.

Download [TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to your desired directory. 

For evaluation on more benchmarks, please refer to [official guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to prepare data and run evaluation.

## Run quantization
Run following codes for LlaVA
```sh
python examples/pytorch/multimodal/quantization/ptq/run_llava_no_trainer.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/coco/images/train2017/ \
    --question-file /path/to/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --quantize \
    --approach weight_only \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_true_sequential \
    --gptq_multimodal \
    --eval-question-file /path/to/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --eval-image-folder /path/to/textvqa/train_images \
    --eval-annotation-file /path/to/textvqa/TextVQA_0.5.1_val.json \
    --eval-result-file ./llava-1.5-7b-hf.jsonl
```

We currently also support quantization for Qwen-VL model
```
python examples/pytorch/multimodal/quantization/ptq/run_llava_no_trainer.py \
    --model_name_or_path Qwen/Qwen-VL \
    --image-folder /path/to/coco/images/train2017/ \
    --question-file /path/to/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --quantize \
    --approach weight_only \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_true_sequential \
    --gptq_nsamples 16 \
    --gptq_multimodal \
    --eval-result-file ./qwen-vl.jsonl

```