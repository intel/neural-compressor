#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

''' 
Script to download and save dataset
'''
from datasets import load_dataset
from argparse import ArgumentParser
import os

def main():
    arg_parser = ArgumentParser(description="Download and save dataset")
    arg_parser.add_argument("-p", "--path_to_save_dataset", type=str,
                             help="path to save the dataset",
                             default="./")
    args = arg_parser.parse_args()
    dataset = load_dataset("glue", "sst2", split= "validation")
    path = os.path.join(args.path_to_save_dataset, "sst2_validation_dataset")
    dataset.save_to_disk(path)
    print("Dataset saved in location: {}".format(path))

if __name__ == "__main__":
    main()
