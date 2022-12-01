#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import shutil
import yaml
from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import model_conversion

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.python.framework import graph_util


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
        device: cpu
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def prepare_dataset():
    
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels

def prepare_model(model_out_path, train_images, train_labels):
    
    # Define the model architecture.
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    
    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(
      train_images,
      train_labels,
      epochs=1,
      validation_split=0.1,
    )

    model.save(model_out_path)

@unittest.skipIf(tf.version.VERSION < '2.4.0', "Only supports tf 2.4.0 or above")
class TestModelConversion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self._baseline_temp_path = './temp_baseline'
        self._qat_temp_path = './temp_qat'
        build_fake_yaml()
        train_images, train_labels = prepare_dataset()
        prepare_model(self._baseline_temp_path, train_images, train_labels)

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        shutil.rmtree(self._qat_temp_path, ignore_errors=True)
        shutil.rmtree(self._baseline_temp_path, ignore_errors=True)

    def test_model_conversion(self):
        from neural_compressor.experimental import ModelConversion, common
        from neural_compressor.conf.config import Conf
        conversion = ModelConversion()
        q_model = conversion.fit(model)
        assert isinstance(q_model, tf.keras.Model)
        conf = Conf('fake_yaml.yaml')
        conversion = ModelConversion(conf)
        q_model = conversion.fit(model)
        assert isinstance(q_model, tf.keras.Model)
        conversion = ModelConversion('fake_yaml.yaml')
        q_model = conversion.fit(model)
        assert isinstance(q_model, tf.keras.Model)

if __name__ == "__main__":
    unittest.main()
