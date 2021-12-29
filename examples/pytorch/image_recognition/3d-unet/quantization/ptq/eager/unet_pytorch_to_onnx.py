# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import onnx
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnunet.training.model_restore import load_model_and_checkpoint_files

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default="build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
                        help="Path to the PyTorch model")
    parser.add_argument("--output_name",
                        default="224_224_160.onnx",
                        help="Name of output model")
    parser.add_argument("--dynamic_bs_output_name",
                        default="224_224_160_dyanmic_bs.onnx",
                        help="Name of output model")
    parser.add_argument("--output_dir",
                        default="build/model",
                        help="Directory to save output model")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    print("Converting PyTorch model to ONNX...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = "./{}/{}".format(args.output_dir, args.output_name)
    dynamic_bs_output_path = "./{}/{}".format(args.output_dir, args.dynamic_bs_output_name)

    print("Loading Pytorch model...")
    checkpoint_name = "model_final_checkpoint"
    folds = 1
    trainer, params = load_model_and_checkpoint_files(args.model_dir, folds, fp16=False, checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[0], False)
    height = 224
    width = 224
    depth = 160
    channels = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.rand([1, channels, height, width, depth]).float().to(device)
    torch.onnx.export(trainer.network, dummy_input, output_path, opset_version=11,
                      input_names=['input'], output_names=['output'])
    torch.onnx.export(trainer.network, dummy_input, dynamic_bs_output_path, opset_version=11,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=({"input": {0: "batch_size"}, "output": {0: "batch_size"}}))

    print("Successfully exported model {} and {}".format(output_path, dynamic_bs_output_path))

if __name__ == "__main__":
    main()
