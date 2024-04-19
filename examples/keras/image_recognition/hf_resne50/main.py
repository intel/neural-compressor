from neural_compressor.tensorflow.utils import BaseDataLoader
import tensorflow as tf
from transformers import AutoImageProcessor
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
input_data = image_processor(image, return_tensors="tf")

class Dataset(object):
    def __init__(self, batch_size=100):
        self.length = 100
        self.batch_size = 1
        self.data = [input_data['pixel_values'].numpy()]*100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], None


calib_dataloader = BaseDataLoader(dataset=Dataset()) 

from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import set_random_seed
set_random_seed(9527)
config = PostTrainingQuantConfig(backend='itex', 
    calibration_sampling_size=[100])
q_model = fit(
    model="resnet50-saved-model/saved_model/1",
    conf=config,
    calib_dataloader=calib_dataloader,
    eval_func=evaluate)
q_model.save("resnet50_uniform_qdq")