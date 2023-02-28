import unittest
import shutil
from pkg_resources import parse_version


def train_func():
    import tensorflow as tf
    from tensorflow import keras
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


class Dataset(object):
    def __init__(self, batch_size=100):
        from tensorflow import keras
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
    import tensorflow as tf
    @classmethod
    def setUpClass(self):
        train_func()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('baseline_model',ignore_errors=True)
        shutil.rmtree('trained_qat_model',ignore_errors=True)

    @unittest.skipIf(parse_version(tf.version.VERSION) < parse_version('2.3.0'), "version check")
    def test_qat(self):
        import tensorflow as tf
        from tensorflow import keras
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        from neural_compressor import training, QuantizationAwareTrainingConfig
        config = QuantizationAwareTrainingConfig()
        compression_manager = training.prepare_compression('./baseline_model', config)
        compression_manager.callbacks.on_train_begin()

        q_aware_model = compression_manager.model.model
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

        compression_manager.callbacks.on_train_end()
        compression_manager.save("trained_qat_model")

if __name__ == '__main__':
    unittest.main()
