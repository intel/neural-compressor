import unittest
import os
import shutil
from tensorflow import keras
import numpy as np

def build_sequential_model():

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Create Keras model
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28), name="input"),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax", name="output")
    ])

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    # Train model
    model.fit(x={"input": train_images}, y={"output": train_labels}, epochs=1)
    model.save("./models/saved_model")

    return

class Dataset(object):
    def __init__(self):
        (train_images, train_labels), (test_images,
                    test_labels) = keras.datasets.fashion_mnist.load_data()
        self.test_images = test_images.astype(np.float32) / 255.0
        self.labels = test_labels

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

    def update(self, predict, label):
        self.pred_list.extend(np.argmax(predict, axis=1))
        self.label_list.extend(label)
        self.samples += len(label) 

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def result(self):
        correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples

class TestMixedPrecisionWithKerasModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ['FORCE_FP16'] = '1'
        os.environ['FORCE_BF16'] = '1'
        build_sequential_model()

    @classmethod
    def tearDownClass(self):
        del os.environ['FORCE_FP16']
        del os.environ['FORCE_BF16']
        shutil.rmtree("./models", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_mixed_precision_with_keras_model(self):
        from neural_compressor.data import DataLoader
        dataset = Dataset()
        dataloader = DataLoader(framework='tensorflow', dataset=dataset)

        from neural_compressor.config import MixedPrecisionConfig
        from neural_compressor import mix_precision
        config = MixedPrecisionConfig()
        q_model = mix_precision.fit(
            model='./models/saved_model',
            config=config,
            eval_dataloader=dataloader, 
            eval_metric=MyMetric())

        # Optional, run quantized model
        import tensorflow as tf
        with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
            tf.compat.v1.import_graph_def(q_model.graph_def, name='')
            out = sess.run(['Identity:0'], feed_dict={'input:0':dataset.test_images})
            print("Inference is done.")

        found_cast = False
        for i in q_model.graph_def.node:
            if i.op == 'Cast':
                found_cast = True
                break
        self.assertEqual(found_cast, True)

if __name__ == "__main__":
    unittest.main()
