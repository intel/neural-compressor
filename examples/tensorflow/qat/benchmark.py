import tensorflow as tf
from tensorflow import keras
import numpy as np

class dataloader(object):
    def __init__(self, batch_size=100):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0 
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.batch_size = batch_size
        self.i = 0

    def __iter__(self):
        while self.i < len(self.test_images):
            yield self.test_images[self.i: self.i + self.batch_size], self.test_labels[self.i: self.i + self.batch_size]
            self.i = self.i + self.batch_size

from lpot.experimental import Benchmark, common
evaluator = Benchmark('mnist.yaml')
evaluator.model = common.Model('quantized_model')
evaluator.b_dataloader = dataloader()
evaluator('accuracy')
