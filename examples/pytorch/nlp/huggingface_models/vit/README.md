## ViT

We leverage HuggingFace Transformers Image Classification [example code](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py). Finetuned ViT models can be found in HuggingFace [model hub](https://huggingface.co/models?other=vit). ```--model_name_or_path``` argument can be changed to any one of the finetuned models. ```--dataset_name``` needs to be changed in accordance with which dataset the model has been finetuned on.

### Run the script

```
python -m neural_coder run_image_classification.py
    --model_name_or_path nateraw/vit-base-beans
    --dataset_name beans
    --remove_unused_columns False
    --do_eval
    --output_dir result
```

By default it runs dynamic quantization. To run static quantization, run with ```python -m neural_coder --approach static```.
