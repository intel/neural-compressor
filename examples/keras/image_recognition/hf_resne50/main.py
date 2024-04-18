from neural_compressor.tensorflow import quantize_model, StaticQuantConfig, Model
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


def weight_name_mapping(name):
    return name

calib_dataloader = BaseDataLoader(dataset=Dataset())
quant_config = StaticQuantConfig()    

model = Model("resnet50-saved-model/saved_model/1", modelType="llm_saved_model")
model.weight_name_mapping = weight_name_mapping

q_model = quantize_model(model, quant_config, calib_dataloader)
q_model.save("resnet50_uniform_qdq")

TFSMlayer = tf.keras.layers.TFSMLayer("resnet50_uniform_qdq", call_endpoint="serving_default")
inputs = tf.keras.Input(shape=(3, 224, 224))
outputs = TFSMlayer(inputs)
model = tf.keras.Model(inputs, outputs)

model.save("quantized_resnet50.keras")
model.summary()

preds = model.predict(input_data['pixel_values'])
print(preds)