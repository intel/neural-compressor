import unittest
import shutil
import tensorflow as tf
from tensorflow import keras


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

    model.save("baseline_model")

class TestTensorflowPruning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        train_func()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('baseline_model',ignore_errors=True)
    
    @unittest.skipIf(tf.version.VERSION < '2.3.0', " keras model need tensorflow version >= 2.3.0, so the case is skipped")
    def test_pruning_utility(self):
        from neural_compressor.experimental import common
        pruning_model = common.Model("baseline_model")
        all_weights_name = pruning_model.get_all_weight_names()
        df, sparsity = pruning_model.report_sparsity()
        self.assertEqual(all_weights_name, [1, 4])
        self.assertEqual(df.empty, False)
        self.assertNotEqual(sparsity, None)

if __name__ == '__main__':
    unittest.main()
