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
import os
import random

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="hpu", help="the device to use", choices=["cpu", "cuda", "hpu"])
parser.add_argument("--real_images_path", type=str, help="path to real images")
parser.add_argument("--diff_images_path", type=str, help="path to images generated from diffusion")
parser.add_argument("--num_of_images", type=int, help="number of images to evaluate with")
opt = parser.parse_args()

real_images_path = opt.real_images_path
diff_images_path = opt.diff_images_path
num_of_images = opt.num_of_images
device = opt.device
if device == "hpu":
    import habana_frameworks.torch.core as core

cosine_sim = torch.nn.CosineSimilarity(dim=0)
fid = FrechetInceptionDistance(feature=2048).to(device)

real_images_path = real_images_path + "/*"
diff_images_path = diff_images_path + "/*"
files = glob.glob(diff_images_path)
files_real = glob.glob(real_images_path)

files = random.sample(files, num_of_images)
files_real = random.sample(files_real, num_of_images)


print("started evaluation")
# Load clip processor and model
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# Calculate Cosine Distance
print("finding clip score")
# Tensor with distance of each image to its label - to be averaged
# distances = torch.Tensor([]).to(device)
distances = []
diff_img_tensor = torch.Tensor([]).to(device)
# images must be 299*299 for FID calculation
transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((299, 299))])

with torch.no_grad():
    for i, curr_image_path in tqdm(enumerate(files)):
        # For clip score
        images = Image.open(curr_image_path)

        # get caption
        text = os.path.basename(curr_image_path)
        text = os.path.splitext(text)[0]

        # process image and text, and embed
        image_processed = processor(images=images, return_tensors="pt").to(device)
        image_embedding = torch.squeeze(clip.get_image_features(**image_processed))
        text_processed = processor(text=text, return_tensors="pt").to(device)
        text_embedding = torch.squeeze(clip.get_text_features(**text_processed))

        # calculate cosine distance for ith image
        cosine_dist = torch.Tensor([cosine_sim(image_embedding, text_embedding)]).to(device)
        distances.append(cosine_dist)
        # distances = torch.cat((distances,cosine_dist), dim = 0)

        # for FID - register these images as real images
        reshaped_img = torch.unsqueeze(transform(images), dim=0).to(device)
        fid.update(reshaped_img.to(dtype=torch.uint8), real=False)

    distance = torch.Tensor(len(distances)).to(device)
    torch.cat(distances, out=distance)
    print(f"mean cosine distance is {torch.mean(distance)}")
    print("finding FID score")

    # add real images to FID calculation
    real_images_tensor = torch.Tensor([]).to(device)

    resize_transform = transforms.Resize((299, 299))
    for i, curr_image_path in tqdm(enumerate(files_real)):
        images = Image.open(curr_image_path)
        real_images = torch.unsqueeze(transform(images), dim=0).to(device)
        if real_images.shape[1:] == (3, 299, 299):
            fid.update(real_images.to(dtype=torch.uint8), real=True)

    print(f"The FID is {fid.compute()}")
