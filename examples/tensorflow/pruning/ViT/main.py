import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from neural_compressor.experimental import Pruning, common
from neural_compressor.utils import logger


# Prepare dataset
def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    logger.info(f"Training set: x_shape-{x_train.shape}, y_shape-{y_train.shape}")
    logger.info(f"Test set: x_shape-{x_test.shape}, y_shape-{y_test.shape}")
    return TrainDataset(x_train, y_train), EvalDataset(x_test, y_test)

# Build TrainDataset and EvalDataset
class TrainDataset(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class EvalDataset(object):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.x_test[idx], self.y_test[idx]


if __name__ == '__main__':
    prune = Pruning("./prune_vit.yaml")
    # prune.train_distributed = True
    # prune.evaluation_distributed = True
    training_set, test_set = prepare_dataset()
    prune.train_dataloader = common.DataLoader(training_set, batch_size=128)
    prune.eval_dataloader = common.DataLoader(test_set, batch_size=256)
    prune.model = './ViT_Model'
    model = prune.fit()
    stats, sparsity = model.report_sparsity()
    logger.info(stats)
    logger.info(sparsity)
