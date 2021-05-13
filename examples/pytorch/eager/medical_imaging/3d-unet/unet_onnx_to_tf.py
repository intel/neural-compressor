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
import onnx_tf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model",
                        default="build/model/224_224_160.onnx",
                        help="Path to the ONNX model")
    parser.add_argument("--output_name",
                        default="224_224_160.pb",
                        help="Name of output model")
    parser.add_argument("--output_dir",
                        default="build/model",
                        help="Directory to save output model")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    print("Loading ONNX model...")
    onnx_model = onnx.load(args.onnx_model)

    print("Converting ONNX model to TF...")
    tf_model = onnx_tf.backend.prepare(onnx_model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = "./{}/{}".format(args.output_dir, args.output_name)

    tf_model.export_graph(output_path)

    print("Successfully exported model {}".format(output_path))


if __name__ == "__main__":
    main()
