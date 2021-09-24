import unittest
import os
import yaml
import shutil
import tensorflow as tf
from tensorflow import keras


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow

        device: cpu
        quantization:
          approach: quant_aware_training
        evaluation:
          accuracy:
            metric:
              Accuracy: {}
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_by_train():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow

        device: cpu
        quantization:
          approach: quant_aware_training
          train:
            optimizer:
              SGD:
                learning_rate: 0.1
            criterion:
              CrossEntropyLoss:
                reduction: none
        evaluation:
          accuracy:
            metric:
              Accuracy: {}
        '''

    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_train.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def train_func():
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_split=0.1,
    )

    _, baseline_model_accuracy = model.evaluate(
        test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    model.save("baseline_model")


def q_func(model):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.load_model("baseline_model")

    import tensorflow_model_optimization as tfmot
    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])

    train_images_subset = train_images[0:1000]  # out of 60000
    train_labels_subset = train_labels[0:1000]

    q_aware_model.fit(train_images_subset, train_labels_subset,
                      batch_size=500, epochs=1, validation_split=0.1)

    _, q_aware_model_accuracy = q_aware_model.evaluate(
        test_images, test_labels, verbose=0)

    print('Quant test accuracy:', q_aware_model_accuracy)
    q_aware_model.save("trained_qat_model")
    return 'trained_qat_model'


class Dataset(object):
    def __init__(self, batch_size=100):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]


class TestTensorflowQAT(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        train_func()
        build_fake_yaml_by_train()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        shutil.rmtree('baseline_model',ignore_errors=True)
        shutil.rmtree('trained_qat_model',ignore_errors=True)
        os.remove('fake_yaml_train.yaml')
    @unittest.skipIf(tf.version.VERSION < '2.3.0', " keras model need tensorflow version >= 2.3.0, so the case is skipped")
    def test_qat_with_external_q_func(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        quantizer.eval_dataloader = common.DataLoader(Dataset())
        quantizer.model = './baseline_model'
        quantizer.q_func = q_func
        quantizer()


if __name__ == '__main__':
    unittest.main()
