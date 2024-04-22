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

from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
from neural_compressor.tensorflow.utils.model_wrappers import TensorflowSavedModelModel

quant_config = StaticQuantConfig()
model = TensorflowSavedModelModel("resnet50-saved-model/saved_model/1")
model.model_type="saved_model"
q_model = quantize_model(model, quant_config, calib_dataloader)

q_model.save("resnet50_uniform_qdq")
