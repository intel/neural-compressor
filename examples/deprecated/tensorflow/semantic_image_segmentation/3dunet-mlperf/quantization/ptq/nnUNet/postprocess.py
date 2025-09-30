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

import numpy as np
from multiprocessing import Pool
import os
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

def load_predictions(predictions, dictionaries, validation_indices):
    assert len(predictions) == len(dictionaries),"Number of predictions does not match number of samples in validation set!"
    padded_shape = [224,224,160]
    results = [None for i in range(len(predictions))]
    for i in range(len(predictions)):
        qsl_idx = validation_indices[i]
        prediction = predictions[qsl_idx]
        assert qsl_idx >= 0 and qsl_idx < len(predictions), "Invalid qsl_idx!"
        raw_shape = list(dictionaries[qsl_idx]["size_after_cropping"])
        # Remove the padded part
        pad_before = [(p - r) // 2 for p, r in zip(padded_shape, raw_shape)]
        pad_after = [-(p - r - b) for p, r, b in zip(padded_shape, raw_shape, pad_before)]
        result_shape = (4,) + tuple(padded_shape)
        result = np.reshape(prediction, result_shape).astype(np.float32)
        results[qsl_idx] = result[:, pad_before[0]:pad_after[0], pad_before[1]:pad_after[1], pad_before[2]:pad_after[2]]
    assert all([i is not None for i in results]), "Missing some results!"
    return results

def postprocess_output(predictions, dictionaries, validation_indices, output_folder, output_files):
    processed_predictions = load_predictions(predictions, dictionaries, validation_indices)
    print("Running postprocessing with multiple threads...")
    force_separate_z=None
    interp_order=3
    interp_order_z=0
    num_threads_nifti_save = 12
    all_in_gpu = "None"
    print("Saving predictions...")
    pool = Pool(num_threads_nifti_save)
    results = []
    for i, output_filename in enumerate(output_files):
        print(i, "/", len(output_files))
        output_filename = os.path.join(output_folder, output_filename + ".nii.gz")
        softmax_mean = processed_predictions[i]
        dct = dictionaries[i]
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax_mean, output_filename, dct, interp_order, None, None, None,
                                            None, None, force_separate_z, interp_order_z),)
                                          ))

    _ = [i.get() for i in results]
    pool.close()
    pool.join()
