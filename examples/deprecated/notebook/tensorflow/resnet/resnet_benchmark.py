import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datasets
from datasets import load_dataset
import argparse
import os

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--input_model", type=str, required=True)
args = parser.parse_args()

# load dataset in streaming way will get an IterableDatset
calib_dataset = load_dataset('imagenet-1k', split='train', streaming=True, use_auth_token=True)
eval_dataset = load_dataset('imagenet-1k', split='validation', streaming=True, use_auth_token=True)

MAX_SAMPLE_LENGTG=1000
def sample_data(dataset, max_sample_length):
    data = {"image": [], "label": []}
    for i, record in enumerate(dataset):
        if i >= MAX_SAMPLE_LENGTG:
            break
        data["image"].append(record['image'])
        data["label"].append(record['label'])
    return datasets.Dataset.from_dict(data)

sub_eval_dataset = sample_data(eval_dataset, MAX_SAMPLE_LENGTG)

from neural_compressor.data.transforms.imagenet_transform import TensorflowResizeCropImagenetTransform
height = width = 224
transform = TensorflowResizeCropImagenetTransform(height, width)


class CustomDataloader:
    def __init__(self, dataset, batch_size=1):
        '''dataset is a iterable dataset and will be loaded record by record at runtime.'''
        self.dataset = dataset
        self.batch_size = batch_size
        import math
        self.length = math.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self):
        batch_inputs = []
        labels = []
        for idx, record in enumerate(self.dataset):
            # record e.g.: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=408x500 ...>, 'label': 91}
            img = record['image']
            label = record['label']
            # skip the wrong shapes
            if len(np.array(img).shape) != 3 or np.array(img).shape[-1] != 3:
                continue
            img_resized = transform((img, label))   # (img, label)
            batch_inputs.append(np.array(img_resized[0]))
            labels.append(label)
            if (idx+1) % self.batch_size == 0:
                yield np.array(batch_inputs), np.array(labels)   # (bs, 224, 224, 3), (bs,)
                batch_inputs = []
                labels = []
    def __len__(self):
        return self.length

from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.utils.create_obj_from_config import create_dataloader

eval_dataloader = CustomDataloader(dataset=sub_eval_dataset, batch_size=1)


from neural_compressor.benchmark import fit
from neural_compressor.config import BenchmarkConfig


conf = BenchmarkConfig(iteration=100,
                       cores_per_instance=4,
                       num_of_instance=1)
bench_dataloader = CustomDataloader(dataset=sub_eval_dataset, batch_size=1)


fit(args.input_model, conf, b_dataloader=bench_dataloader)
