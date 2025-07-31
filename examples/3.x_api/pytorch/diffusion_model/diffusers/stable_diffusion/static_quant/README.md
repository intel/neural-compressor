# Stable Diffusion

Stable Diffusion quantization and inference best known configurations with static quant.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/stabilityai/stable-diffusion-2-1       |           -           |         -          |

# Pre-Requisite

### Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset will be downloaded. This environment variable will be used again when running training scripts.
```
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

# Quantization and Inference
quantization
```shell
python main.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --dataset_path=${DATASET_DIR} \
    --quantized_model_path=${INT8_MODEL} \
    --compile_inductor \
    --precision=int8-bf16 \
    --calibration
```
inference
```shell
python main.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --dataset_path=${DATASET_DIR} \
    --precision=int8-bf16 \
    --benchmark \
    -w 1 \
    -i 10 \
    --quantized_model_path=${INT8_MODEL} \ --compile_inductor
```
## FID evaluation
We have also evaluated FID scores on COCO2017 validation dataset for BF16 model, mixture of BF16 and INT8 model. FID results are listed below.

| Model                | BF16  | INT8+BF16 |
|----------------------|-------|-----------|
| stable-diffusion-2-1 | 27.94 |  27.14    |

To evaluated FID score on COCO2017 validation dataset for mixture of BF16 and INT8 model, you can use below command.

```bash
python main.py \
    --dataset_path=${DATASET_DIR} \
    --precision=int8-bf16 \
    --accuracy \
    --quantized_model_path=${INT8_MODEL} \ --compile_inductor
```