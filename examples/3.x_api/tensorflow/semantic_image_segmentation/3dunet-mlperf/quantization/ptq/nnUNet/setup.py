#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

import os, shutil
import argparse
import sys
import zipfile
#import sys
#print(sys.path)
#sys.path.append('/home/sys_dltest/lpot/lz/frameworks.ai.models.intel-models/models/image_segmentation/tensorflow/3d_unet_mlperf')
from nnUNet.Task043_BraTS_2019 import task_setup
from nnUNet.preprocess import preprocess_setup

BUILD_DIR = 'build'
RAW_DATA_DIR = BUILD_DIR + '/raw_data'
PREPROCESSED_DATA_DIR = BUILD_DIR + '/preprocessed_data'
POSTPROCESSED_DATA_DIR = BUILD_DIR + '/postprocessed_data'
MODEL_DIR = BUILD_DIR + '/model'
RESULT_DIR = BUILD_DIR + '/result'
TF_MODEL = '224_224_160.pb'
OTHER_FILES = 'fold_1.zip'

def create_directories():
    print("Creating directories")
    if not os.path.isdir(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    if not os.path.isdir(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)
    if not os.path.isdir(POSTPROCESSED_DATA_DIR):
        os.makedirs(POSTPROCESSED_DATA_DIR)
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def download_model(input_graph):
    pwd = os.getcwd()
    os.chdir(os.path.join(pwd, MODEL_DIR))
    if input_graph == 'NONE':
        print("Downloading TF model from Zenodo")
        if not os.path.isfile(TF_MODEL):
            os.system('wget -O 224_224_160.pb https://zenodo.org/record/3928991/files/224_224_160.pb?download=1;')
    os.chdir(os.path.join(pwd, RESULT_DIR))
    if not os.path.isfile(OTHER_FILES):
        os.system('wget -O fold_1.zip https://zenodo.org/record/3904106/files/fold_1.zip?download=1;')
        zip_file = "fold_1.zip"
        #legacy bitmap issue https://bugzilla.redhat.com/show_bug.cgi?id=1802689
        if (not os.path.isfile(OTHER_FILES)):
            os.system('curl -O --output fold_1.zip https://zenodo.org/record/3904106/files/fold_1.zip')
        try:
            with zipfile.ZipFile(zip_file) as z:
                z.extractall()
                print("Extracted all")
        except:
            print("Could not extract fold_1.zip")
    os.chdir(pwd)

def setup(downloaded_data_dir, input_graph='NONE'):
    create_directories()
    download_model(input_graph)
    task_setup(downloaded_data_dir)
    preprocess_setup(PREPROCESSED_DATA_DIR)
