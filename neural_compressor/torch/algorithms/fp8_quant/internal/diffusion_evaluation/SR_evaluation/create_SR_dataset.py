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
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.utils import save_image


class CenterCropAndResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        crop_size = min(width, height)
        crop = F.center_crop(img, (crop_size, crop_size))
        resize = F.resize(crop, self.size)
        return resize


def get_data_loader(path, dataset="ImageNet", workers=4, shuffle=None, pin_memory=True, resize=256):

    # Data loader for ImageNet data.

    # defines desired resize amd creates dataset
    def get_dataset(path_to_data):
        transformations = [CenterCropAndResize(resize), transforms.ToTensor()]
        return datasets.ImageFolder(path_to_data, transforms.Compose(transformations))

    # checks if given path is valid
    if isinstance(path, str):
        curr_path = path
        if not os.path.exists(curr_path):
            raise FileNotFoundError(f"Directory {curr_path} doesn't exist")
        data_dir = curr_path
    elif isinstance(path, list):
        for path_ in path:
            if os.path.exists(path_):
                curr_path = path_
                break
        else:
            raise FileNotFoundError(
                "None of the default data directories exist in your env," " please manually specify one"
            )
        data_dir = os.path.join(curr_path, "val")
    else:
        raise ValueError("get_data_loader expects list of paths or single path")

    # create dataloader from dataset
    dataset = get_dataset(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle, num_workers=workers, pin_memory=pin_memory
    )

    return data_loader


parser = argparse.ArgumentParser("Create dataset of real images for SR evaluation", add_help=False)

parser.add_argument("--images", type=str, help="path to imagenet validation set")
parser.add_argument(
    "--out_dir", type=str, help="path to save images with correct format (cropped + modified file name)"
)
parser.add_argument("--resize", type=int, default=256, help="dimensions to resize image")
parser.add_argument(
    "--class_to_labels",
    type=str,
    default="imagenet1000_clsidx_to_labels.txt",
    help="path to text file containing" "mapping between class index and label",
)

args = parser.parse_args()
images = args.images
out_dir = args.out_dir
resize = args.resize
class_to_labels = args.class_to_labels

with torch.no_grad():
    # get dataloader
    dl = get_data_loader(images, resize=resize)

    # open idx2label, which matches an integer signifying class with the corrects label
    idx2label = eval(open(class_to_labels).read())

    # save images with correct filename
    for i, image in enumerate(dl):
        label = idx2label.get(image[1].item())
        save_image(image[0], f"{out_dir}/{label}_{i}.png")
