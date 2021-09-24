import tensorflow
from  tensorflow.keras.datasets import cifar10

from tensorflow import keras
import numpy as np

num_classes = 10
class EvalDataset(object):
    def __init__(self, batch_size=100):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        # Convert class vectors to binary class matrices.
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
        self.test_images = x_test
        self.test_labels = y_test

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]


from neural_compressor.experimental import Benchmark, common
evaluator = Benchmark('benchmark.yaml')
evaluator.model = common.Model('./pruned_model')
evaluator.b_dataloader = common.DataLoader(EvalDataset())
evaluator('performance')
