#!/usr/bin/env python
# coding=utf-8
# export pytorch model into onnx model
import argparse
import os
import subprocess


def get_dummy_input(model_name_or_path):
    import torch
    from PIL import Image
    from transformers import LayoutLMv2Processor

    processor = LayoutLMv2Processor.from_pretrained(model_name_or_path)

    width = 762
    height = 800
    # Create a new RGB image with the specified size
    image = Image.new("RGB", (width, height))
    # Generate random pixel values and set them in the image
    pixels = []
    import numpy as np

    red = np.random.randint(255, size=(width * height))
    green = np.random.randint(255, size=(width * height))
    blue = np.random.randint(255, size=(width * height))

    pixels = [(r, g, b) for r, g, b in zip(red, green, blue)]

    image.putdata(pixels)

    encoding = processor(image, return_tensors="pt", max_length=512, padding="max_length")
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
    for key, val in encoding.items():
        print(f"key: {key}; val: {val.shape}")
    dummy_input = {}
    dummy_input["input_ids"] = encoding["input_ids"].to(torch.int64)
    dummy_input["attention_mask"] = encoding["attention_mask"].to(torch.int64)
    dummy_input["bbox"] = encoding["bbox"].to(torch.int64)
    dummy_input["image"] = encoding["image"].to(torch.int64)
    # image torch.Size([4, 3, 224, 224])
    # input_ids torch.Size([4, 512])
    # attention_mask torch.Size([4, 512])
    # token_type_ids torch.Size([4, 512])
    # bbox torch.Size([4, 512, 4])
    # labels torch.Size([4, 512])
    return dummy_input


def export_model_to_onnx(model_name_or_path, export_model_path):
    from collections import OrderedDict
    from itertools import chain

    from torch.onnx import export as onnx_export
    from transformers import LayoutLMv2ForTokenClassification

    # labels = datasets['train'].features['ner_tags'].feature.names
    # id2label = {v: k for v, k in enumerate(labels)}
    # label2id = {k: v for v, k in enumerate(labels)}
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_name_or_path, num_labels=7)
    dummy_input = get_dummy_input(model_name_or_path)
    inputs = OrderedDict(
        {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "image": {0: "batch_size", 1: "num_channels"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
    )
    assert len(inputs.keys()) == len(dummy_input.keys())
    outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})

    onnx_export(
        model=model,
        args=(dummy_input,),
        f=export_model_path,
        input_names=list(inputs.keys()),
        output_names=list(outputs.keys()),
        dynamic_axes=dict(chain(inputs.items(), outputs.items())),
        do_constant_folding=True,
    )
    assert os.path.exists(export_model_path), f"{export_model_path} doesn't exist!"
    print(f"The model was successfully exported and saved as {export_model_path}.")


def prepare_env():
    subprocess.run(
        ["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"],
        stdout=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export huggingface onnx model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model",
                        type=str,
                        default="nielsr/layoutlmv2-finetuned-funsd",
                        help="fine-tuned pytorch model name or path")
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--max_len",
                        type=int,
                        default=512,
                        help="Maximum length of the sentence pairs")
    args = parser.parse_args()
    prepare_env()
    export_model_to_onnx(model_name_or_path=args.input_model, export_model_path=args.output_model)
