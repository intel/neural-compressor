# Copyright (c) 2024 Intel Corporation
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

import argparse
import glob

# load data
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", type=str, help="path to original dataset")
parser.add_argument("--save_path", type=str, help="path to save images to")
parser.add_argument("--num_of_images", type=int, help="number of images to save")
parser.add_argument("--num_of_gpus", type=int, default=1, help="split csv to num_of_gpu files, to train in parallel")
opt = parser.parse_args()

dataset_path = opt.images_path
num_of_gpus = opt.num_of_gpus
save_path = opt.save_path
n_samples = opt.num_of_images


annotations_path = dataset_path + "/annotations/captions_val2017.json"
images_path = dataset_path + "/val2017/"

annotations_file = json.load(open(annotations_path))

if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path_images = save_path + "/subset"


# merge images and annotations
images = annotations_file["images"]
annotations = annotations_file["annotations"]
df = pd.DataFrame(images)
df_annotations = pd.DataFrame(annotations)
df = df.merge(pd.DataFrame(annotations), how="left", left_on="id", right_on="image_id")

# keep only the relevant columns
df = df[["file_name", "caption"]]


# shuffle the dataset
df = df.sample(frac=1)

# remove duplicate images
df = df.drop_duplicates(subset="file_name")

# sample from data, remove slashes (because prompts with slashes are a problem for SD)
df_sample = df.sample(n_samples)
df_sample = df_sample[~df_sample.iloc[:, 1].str.contains("/")]


num_per_file = int(np.floor(df_sample.shape[0] / num_of_gpus))
for i in range(num_of_gpus):
    # save captions to csv
    # rename old csv
    save_path_csv = save_path + f"/subset_{i}.csv"
    if os.path.exists(save_path_csv):
        num_csv = len(glob.glob(save_path + "/*.csv"))
        os.rename(save_path_csv, save_path + f"/old_{num_csv}_subset_{i-num_of_gpus}.csv")
        df_sample["caption"].iloc[num_per_file * i + 1 : num_per_file * (1 + i) + 1].to_csv(
            save_path_csv, index=False, header=False
        )
        print("Already found a csv named 'subset'")
        print(f"Renamed it as '/old_{num_csv}_subset.csv")

    else:
        df_sample["caption"].iloc[num_per_file * i : num_per_file * (1 + i)].to_csv(
            save_path_csv, index=False, header=False
        )

print("Saved your new csv to " + save_path)

# clean folder of images
files = glob.glob(save_path_images + "/*.jpg")
if os.path.exists(save_path_images):
    print("Already found a folder named subset")
    num_subsets = len(list(os.walk(save_path)))
    os.rename(save_path_images, save_path + f"/old_{num_subsets}_subset")
    print(f"Renamed it as '/old_{num_subsets}_subset")


# copy the images to reference folder
subset_path = Path(save_path_images)
subset_path.mkdir(exist_ok=True)
for i, row in df_sample.iterrows():
    path = images_path + row["file_name"]
    shutil.copy(path, subset_path)

print("saved your images to " + save_path_images)
