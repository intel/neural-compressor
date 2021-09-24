#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import shutil
import yaml
import tensorflow as tf

from lpot.experimental import model_conversion
tf.compat.v1.enable_eager_execution()
from tensorflow import keras

from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
        device: cpu
        model_conversion:
          source: qat
          destination: default
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

def prepare_qat_model(model_in_path, model_out_path, train_images, train_labels):
    import tensorflow_model_optimization as tfmot
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # q_aware stands for for quantization aware.
    model = tf.keras.models.load_model(model_in_path)
    q_aware_model = quantize_model(model)
    
    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    train_images_subset = train_images[0:1000] # out of 60000
    train_labels_subset = train_labels[0:1000]
    
    q_aware_model.fit(train_images_subset, train_labels_subset,
                      batch_size=500, epochs=1, validation_split=0.1)
    
    q_aware_model.save(model_out_path)

@unittest.skipIf(tf.version.VERSION < '2.4.0', "Only supports tf 2.4.0 or above")
class TestModelConversion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self._baseline_temp_path = './temp_baseline'
        self._qat_temp_path = './temp_qat'
        self._quantized_temp_path = './temp_quantized'
        build_fake_yaml()
        train_images, train_labels = prepare_dataset()
        prepare_model(self._baseline_temp_path, train_images, train_labels)
        prepare_qat_model(self._baseline_temp_path, self._qat_temp_path, train_images, train_labels)

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        shutil.rmtree(self._qat_temp_path, ignore_errors=True)
        shutil.rmtree(self._baseline_temp_path, ignore_errors=True)
        shutil.rmtree(self._quantized_temp_path, ignore_errors=True)

    def test_model_conversion(self):
        from neural_compressor.experimental import ModelConversion, common
        from neural_compressor.conf.config import Conf
        conversion = ModelConversion()
        conversion.source = 'qat'
        conversion.destination = 'default'
        conversion.model = common.Model(self._qat_temp_path)
        q_model = conversion()
        q_model.save(self._quantized_temp_path)
        conf = Conf('fake_yaml.yaml')
        conversion = ModelConversion(conf)
        conversion.source = 'qat'
        conversion.destination = 'default'
        conversion.model = common.Model(self._qat_temp_path)
        q_model = conversion()
        conversion = ModelConversion('fake_yaml.yaml')
        conversion.source = 'qat'
        conversion.destination = 'default'
        conversion.model = common.Model(self._qat_temp_path)
        q_model = conversion()

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                meta_graph=tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], self._quantized_temp_path)
                print(meta_graph.graph_def.node)
                for i in meta_graph.graph_def.node:
                    if 'MatMul' in i.op:
                        self.assertTrue('QuantizedMatMul' in i.op)
                    if 'MaxPool' in i.op:
                        self.assertTrue('QuantizedMaxPool' in i.op)
                    if 'Conv2D' in i.op:
                        self.assertTrue('QuantizedConv2D' in i.op)

if __name__ == "__main__":
    unittest.main()
