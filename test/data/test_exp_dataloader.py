"""Tests for the dataloader module."""
import os
import platform
import shutil
import unittest

import numpy as np
from PIL import Image

from neural_compressor.experimental.data import DATALOADERS, TRANSFORMS, Datasets
from neural_compressor.utils.create_obj_from_config import create_dataloader, create_dataset


class TestDataloader(unittest.TestCase):
    def test_iterable_dataset(self):
        class iter_dataset(object):
            def __iter__(self):
                for i in range(100):
                    yield np.zeros([256, 256, 3])

        dataset = iter_dataset()
        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))

    def test_tensorflow_dummy(self):
        datasets = Datasets("tensorflow")
        dataset = datasets["dummy"](shape=(4, 256, 256, 3))

        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))

        with self.assertRaises(AssertionError):
            dataset = datasets["dummy"](shape=[(4, 256, 256, 3), (256, 256, 3)])
        with self.assertRaises(AssertionError):
            dataset = datasets["dummy"](shape=(4, 256, 256, 3), low=[1.0, 0.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["dummy"](shape=(4, 256, 256, 3), high=[128.0, 127.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["dummy"](shape=(4, 256, 256, 3), dtype=["float32", "int8"])

    def test_tensorflow_dummy_v2(self):
        datasets = Datasets("tensorflow")
        # test with label
        dataset = datasets["dummy_v2"](input_shape=(256, 256, 3), label_shape=(1,))
        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        self.assertEqual(data[1].shape, (1, 1))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))
        self.assertEqual(data[1].shape, (2, 1))

        # test without label
        dataset = datasets["dummy_v2"](input_shape=(256, 256, 3))
        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

        with self.assertRaises(AssertionError):
            dataset = datasets["dummy_v2"](input_shape=(256, 256, 3), low=[1.0, 0.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["dummy_v2"](input_shape=(256, 256, 3), high=[128.0, 127.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["dummy_v2"](input_shape=(256, 256, 3), dtype=["float32", "int8"])

    def test_tensorflow_sparse_dummy_v2(self):
        datasets = Datasets("tensorflow")
        # test with label
        dataset = datasets["sparse_dummy_v2"](
            dense_shape=[[10, 20], [5, 3]], label_shape=[[1]], sparse_ratio=[0.98, 0.8]
        )
        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0][0][0].shape, (1, 4, 2))
        self.assertEqual(data[0][0][1].shape, (1, 4))
        self.assertEqual(data[0][1].shape, (1, 1))

        # test without label
        dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), sparse_ratio=0.3)
        data_loader = DATALOADERS["tensorflow"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0][0].shape, (1, 137626, 3))
        self.assertEqual(data[0][1].shape, (1, 137626))

        with self.assertRaises(AssertionError):
            dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), low=[1.0, 0.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), high=[128.0, 127.0])
        with self.assertRaises(AssertionError):
            dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), dtype=["float32", "int8"])
        with self.assertRaises(AssertionError):
            dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), dtype=["0.3", "0.5"])
        with self.assertRaises(AssertionError):
            dataset = datasets["sparse_dummy_v2"](dense_shape=(256, 256, 3), label_shape=[[1], [2], [3]])

    def test_tensorflow_dataloader_multi_input(self):
        import tensorflow as tf

        x = tf.data.Dataset.from_tensor_slices((np.random.random(20), np.random.random(20)))
        y = tf.data.Dataset.from_tensor_slices(np.random.random(20))
        dataset = tf.data.Dataset.zip((x, y))
        dataloader = DATALOADERS["tensorflow"](dataset)
        for i, (x, y) in enumerate(dataloader):
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            break

    def test_style_transfer_dataset(self):
        random_array = np.random.random_sample([100, 100, 3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save("test.jpg")

        datasets = Datasets("tensorflow")
        dataset = datasets["style_transfer"](content_folder="./", style_folder="./")
        length = len(dataset)
        image, label = dataset[0]
        self.assertEqual(image[0].shape, (256, 256, 3))
        self.assertEqual(image[1].shape, (256, 256, 3))
        os.remove("test.jpg")

    def test_tensorflow_list_dict(self):
        dataset = [{"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 5, "b": 6, "c": 7, "d": 8}]
        data_loader = DATALOADERS["tensorflow"](dataset)

        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {"a": [1], "b": [2], "c": [3], "d": [4]})

        # test iterable consistent
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {"a": [1], "b": [2], "c": [3], "d": [4]})

        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]})

    def test_pytorch_dummy(self):
        datasets = Datasets("pytorch")
        transform = TRANSFORMS("pytorch", "preprocess")["Resize"](**{"size": 100})
        dataset = datasets["dummy"](
            shape=[(4, 256, 256, 3), (4, 1)], high=[10.0, 10.0], low=[0.0, 0.0], transform=transform
        )

        data_loader = DATALOADERS["pytorch"](dataset)
        iterator = iter(data_loader)
        data, label = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data, label = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))

    @unittest.skipIf(platform.system().lower() == "windows", "not support mxnet on windows yet")
    def test_mxnet_dummy(self):
        datasets = Datasets("mxnet")
        transform = TRANSFORMS("mxnet", "preprocess")["Resize"](**{"size": 100})
        dataset = datasets["dummy"](shape=(4, 256, 256, 3), transform=transform)

        data_loader = DATALOADERS["mxnet"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))

        dataset = datasets["dummy"](shape=(4, 256, 256, 3), label=True)
        self.assertEqual(dataset[0][1], 0)

    def test_onnxrt_qlinear_dummy(self):
        datasets = Datasets("onnxrt_qlinearops")
        transform = TRANSFORMS("onnxrt_qlinearops", "preprocess")["Resize"](**{"size": 100})
        dataset = datasets["dummy"](shape=(4, 256, 256, 3), transform=transform)

        data_loader = DATALOADERS["onnxrt_qlinearops"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))

        dataset = datasets["dummy"](shape=(4, 256, 256, 3), label=False)
        data_loader = DATALOADERS["onnxrt_qlinearops"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(len(data), 1)

        with self.assertRaises(AssertionError):
            dataset = datasets["dummy"](shape=[(4, 256, 256, 3), (4, 256, 256, 3)], dtype=["float32", "int8", "int8"])

    def test_onnx_integer_dummy(self):
        datasets = Datasets("onnxrt_integerops")
        dataset = datasets["dummy"](shape=(4, 256, 256, 3))

        data_loader = DATALOADERS["onnxrt_integerops"](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch="rollover")
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))


if __name__ == "__main__":
    unittest.main()
