from tensorflow import keras
import numpy as np

class Dataset(object):
    def __init__(self):
        (train_images, train_labels), (test_images,
                    test_labels) = keras.datasets.fashion_mnist.load_data()
        self.test_images = test_images.astype(np.float32) / 255.0
        self.labels = test_labels
        pass

    def __getitem__(self, index):
        return self.test_images[index], self.labels[index]

    def __len__(self):
        return len(self.test_images)

# Define a customized Metric function 
from neural_compressor.metric import BaseMetric
class MyMetric(BaseMetric):
    def __init__(self, *args):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
        pass

    def update(self, predict, label):
        self.pred_list.extend(np.argmax(predict, axis=1))
        self.label_list.extend(label)
        self.samples += len(label) 
        pass

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
        pass

    def result(self):
        correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples


# Quantize with customized dataloader and metric
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import DataLoader
dataset = Dataset()
dataloader = DataLoader(framework='tensorflow', dataset=dataset)
config = PostTrainingQuantConfig(backend='itex')
q_model = fit(
    model='../models/saved_model',
    conf=config,
    calib_dataloader=dataloader,
    eval_dataloader=dataloader,
    eval_metric=MyMetric())

# Optional, run quantized model
keras_model = q_model.model
predictions = keras_model.predict_on_batch(dataset.test_images)
print("Inference is done.")
