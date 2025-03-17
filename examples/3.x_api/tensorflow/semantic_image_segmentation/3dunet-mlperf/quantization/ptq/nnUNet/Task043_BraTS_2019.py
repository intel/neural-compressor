#    Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is copied from nnUnet/nnunet/dataset_conversion/Task043_BraTS_2019.py, except that
# the validation/test set part is removed and downloaded_data_dir is now configurable.

import argparse
import numpy as np
from collections import OrderedDict
import os
import sys

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def task_setup(downloaded_data_dir):
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task043_BraTS2019"
    print(task_name)
    print(downloaded_data_dir)
    print(nnUNet_raw_data)

    target_base = join(nnUNet_raw_data, task_name)
    if not os.path.isdir(target_base):
        target_imagesTr = join(target_base, "imagesTr")
        target_imagesVal = join(target_base, "imagesVal")
        target_imagesTs = join(target_base, "imagesTs")
        target_labelsTr = join(target_base, "labelsTr")

        maybe_mkdir_p(target_imagesTr)
        maybe_mkdir_p(target_imagesVal)
        maybe_mkdir_p(target_imagesTs)
        maybe_mkdir_p(target_labelsTr)

        patient_names = []
        for tpe in ["HGG", "LGG"]:
            cur = join(downloaded_data_dir, tpe)
            for p in subdirs(cur, join=False):
                patdir = join(cur, p)
                patient_name = tpe + "__" + p
                patient_names.append(patient_name)
                t1 = join(patdir, p + "_t1.nii.gz")
                t1c = join(patdir, p + "_t1ce.nii.gz")
                t2 = join(patdir, p + "_t2.nii.gz")
                flair = join(patdir, p + "_flair.nii.gz")
                seg = join(patdir, p + "_seg.nii.gz")

                assert all([
                    isfile(t1),
                    isfile(t1c),
                    isfile(t2),
                    isfile(flair),
                    isfile(seg)
                ]), "%s" % patient_name

                shutil.copy(t1, join(target_imagesTr, patient_name + "_0000.nii.gz"))
                shutil.copy(t1c, join(target_imagesTr, patient_name + "_0001.nii.gz"))
                shutil.copy(t2, join(target_imagesTr, patient_name + "_0002.nii.gz"))
                shutil.copy(flair, join(target_imagesTr, patient_name + "_0003.nii.gz"))

                copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))

        json_dict = OrderedDict()
        json_dict['name'] = "BraTS2019"
        json_dict['description'] = "nothing"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "see BraTS2019"
        json_dict['licence'] = "see BraTS2019 license"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing",
            "3": "enhancing",
        }
        json_dict['numTraining'] = len(patient_names)
        json_dict['numTest'] = 0
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                patient_names]
        json_dict['test'] = []

        save_json(json_dict, join(target_base, "dataset.json"))
    print("DONE")
