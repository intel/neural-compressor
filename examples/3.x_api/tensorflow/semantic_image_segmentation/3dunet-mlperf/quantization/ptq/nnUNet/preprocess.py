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

# This file has been copied from 
# https://github.com/mlcommons/inference/blob/r0.7/vision/medical_imaging/3d-unet/preprocess.py

import argparse
import numpy
import os
import pickle
import sys
import torch

from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import subfiles
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.predict import preprocess_multithreaded

def preprocess_MLPerf(model, checkpoint_name, folds, fp16, list_of_lists, output_filenames, preprocessing_folder, num_threads_preprocessing):
    assert len(list_of_lists) == len(output_filenames)
    print("loading parameters for folds", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, fp16, checkpoint_name=checkpoint_name)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, output_filenames, num_threads_preprocessing, None)
    print("Preprocessing images...")
    all_output_files = []

    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed

        all_output_files.append(output_filename)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        # Pad to the desired full volume
        d = pad_nd_image(d, trainer.patch_size, "constant", None, False, None)

        with open(os.path.join(preprocessing_folder, output_filename+ ".pkl"), "wb") as f:
            pickle.dump([d, dct], f)
        f.close()

    return  all_output_files


def preprocess_setup(preprocessed_data_dir):
    print("Preparing for preprocessing data...")

    # Validation set is fold 1
    fold = 1
    import sys
    import os
    CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
    #validation_fold_file = '/workspace/intelai_models/inference/nnUNet/folds/fold1_validation.txt'
    validation_fold_file = os.path.join(CURRENT_DIR, 'folds/fold1_validation.txt')
    # Make sure the model exists
    model_dir = 'build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1'
    model_path = os.path.join(model_dir, "plans.pkl")
    assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
    checkpoint_name = "model_final_checkpoint"

    # Other settings
    fp16 = False
    num_threads_preprocessing = 12
    raw_data_dir = 'build/raw_data/nnUNet_raw_data/Task043_BraTS2019/imagesTr'

    # Open list containing validation images from specific fold (e.g. 1)
    validation_files = []
    with open(validation_fold_file) as f:
        for line in f:
            validation_files.append(line.rstrip())

    # Create output and preprocessed directory
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    # Create list of images locations (i.e. 4 images per case => 4 modalities)
    all_files = subfiles(raw_data_dir, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[os.path.join(raw_data_dir, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in validation_files]

    # Preprocess images, returns filenames list
    # This runs in multiprocess
    print("Actually preprocessing data...")
    
    preprocessed_files = preprocess_MLPerf(model_dir, checkpoint_name, fold, fp16, list_of_lists,
        validation_files, preprocessed_data_dir, num_threads_preprocessing)

    print("Saving metadata of the preprocessed data...")
    with open(os.path.join(preprocessed_data_dir, "preprocessed_files.pkl"), "wb") as f:
        pickle.dump(preprocessed_files, f)

    print("Preprocessed data saved to {:}".format(preprocessed_data_dir))
    print("Done!")
