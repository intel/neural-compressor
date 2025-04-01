"""Tests for the transform module."""

import os
import platform
import random
import unittest

import numpy as np
from PIL import Image

from neural_compressor.data import DATALOADERS, TRANSFORMS
from neural_compressor.utils.create_obj_from_config import create_dataset, get_postprocess
from neural_compressor.utils.utility import LazyImport

tf = LazyImport("tensorflow")
torch = LazyImport("torch")
torchvision = LazyImport("torchvision")

random.seed(1)
np.random.seed(1)


class TestMetrics(unittest.TestCase):
    def test_tensorflow_2(self):
        image = np.ones([256, 256, 1])
        resize_kwargs = {"size": [224, 224]}
        transforms = TRANSFORMS(framework="tensorflow", process="preprocess")
        resize = transforms["Resize"](**resize_kwargs)
        random_crop_kwargs = {"size": 128}
        random_crop = transforms["RandomCrop"](**random_crop_kwargs)
        transform_list = [resize, random_crop]
        compose = transforms["Compose"](transform_list)
        image_result = compose((image, None))
        self.assertEqual(image_result[0].shape, (128, 128))


class TestONNXQLImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([600, 600]) * 255

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS("onnxrt_qlinearops", "preprocess")
        transform = transforms["ResizeCropImagenet"](height=224, width=224, random_crop=True)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)


class TestONNXITImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([600, 600, 3]) * 255

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS("onnxrt_integerops", "preprocess")
        transform = transforms["ResizeCropImagenet"](height=224, width=224)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)

    def testResizeWithAspectRatio(self):
        transforms = TRANSFORMS("onnxrt_integerops", "preprocess")
        transform = transforms["ResizeWithAspectRatio"](height=224, width=224)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 256)
        self.assertEqual(len(resized_input[0]), 256)
        self.assertEqual(len(resized_input[0][0]), 3)


class TestTensorflowImagenetTransform(unittest.TestCase):
    tf.compat.v1.disable_v2_behavior()

    def testBilinearImagenetTransform(self):
        transforms = TRANSFORMS("tensorflow", "preprocess")
        transform = transforms["BilinearImagenet"](height=224, width=224)
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 0)
        result = transform(sample)
        resized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)

        transforms = TRANSFORMS("onnxrt_qlinearops", "preprocess")
        transform = transforms["BilinearImagenet"](height=224, width=224)
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 0)
        result = transform(sample)
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)

    def testResizeCropImagenetTransform1(self):
        transforms = TRANSFORMS("tensorflow", "preprocess")
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 0)
        transform = transforms["ResizeCropImagenet"](
            height=224, width=224, random_crop=True, random_flip_left_right=True
        )
        result = transform(sample)
        resized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)

    @unittest.skipIf(tf.version.VERSION < "2.5.0", "Skip tf.experimental.numpy.moveaxis")
    def testResizeCropImagenetTransform2(self):
        transforms = TRANSFORMS("tensorflow", "preprocess")
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 0)
        transform = transforms["ResizeCropImagenet"](
            height=224,
            width=224,
            random_crop=False,
            random_flip_left_right=False,
            data_format="channels_last",
            subpixels="RGB",
        )
        result = transform(sample)
        resized_input1 = result[0].eval(session=tf.compat.v1.Session())
        transform = transforms["ResizeCropImagenet"](
            height=224,
            width=224,
            random_crop=False,
            random_flip_left_right=False,
            data_format="channels_last",
            subpixels="BGR",
        )
        result = transform(sample)
        resized_input2 = result[0].eval(session=tf.compat.v1.Session())
        self.assertTrue((resized_input1[..., 0] == resized_input2[..., -1]).all())

        transform = transforms["ResizeCropImagenet"](
            height=224,
            width=224,
            random_crop=False,
            random_flip_left_right=False,
            data_format="channels_first",
            subpixels="BGR",
        )
        rand_input = np.moveaxis(rand_input, -1, 0)
        sample = (rand_input, 0)
        result = transform(sample)
        resized_input3 = result[0].eval(session=tf.compat.v1.Session())
        self.assertTrue((resized_input1[..., 0] == resized_input3[..., -1]).all())

    def testLabelShift(self):
        transforms = TRANSFORMS("tensorflow", "postprocess")
        transform = transforms["LabelShift"](label_shift=1)
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 1001)
        label = transform(sample)[1]
        self.assertEqual(label, 1000)
        if platform.architecture()[0] == "64bit":
            self.assertTrue(isinstance(label, np.int64) or isinstance(label, np.int32))
        else:
            self.assertTrue(isinstance(label, np.int32))

        label = transform((rand_input, [(1, 2, 3)]))[1]
        self.assertTrue(isinstance(label, list))
        self.assertTrue(isinstance(label[0], tuple))

        label = transform((rand_input, [[1, 2, 3]]))[1]
        self.assertTrue(isinstance(label, list))
        self.assertTrue(isinstance(label[0], list))

        label = transform((rand_input, [np.array([1, 2, 3])]))[1]
        self.assertTrue(isinstance(label, list))
        self.assertTrue(isinstance(label[0], np.ndarray))

    def testQuantizedInput(self):
        transforms = TRANSFORMS("tensorflow", "preprocess")
        transform = transforms["QuantizedInput"](dtype="uint8", scale=100)
        rand_input = np.random.random_sample([600, 600, 3]).astype(np.float32)
        sample = (rand_input, 1001)
        result = transform(sample)
        quantized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertLessEqual(quantized_input.max(), 255)
        self.assertGreaterEqual(quantized_input.min(), 0)

        transform = transforms["QuantizedInput"](dtype="uint8")
        sample = (rand_input, 1001)
        result = transform(sample)
        quantized_input = result[0]
        self.assertLessEqual(quantized_input.max(), 1)
        self.assertGreaterEqual(quantized_input.min(), 0)


class TestDataConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([10, 10, 3]) * 255
        cls.pt_trans = TRANSFORMS("pytorch", "preprocess")

    def testToPILImage(self):
        trans = TestDataConversion.pt_trans["ToPILImage"]()
        image, _ = trans((TestDataConversion.img.astype(np.uint8), None))
        self.assertTrue(isinstance(image, Image.Image))

    def testToTensor(self):
        trans = TestDataConversion.pt_trans["ToTensor"]()
        image, _ = trans((TestDataConversion.img.astype(np.uint8), None))
        self.assertTrue(isinstance(image, torch.Tensor))


class TestSameTransfoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([10, 10, 3]) * 255
        cls.tf_trans = TRANSFORMS("tensorflow", "preprocess")
        cls.pt_trans = TRANSFORMS("pytorch", "preprocess")
        cls.ox_trans = TRANSFORMS("onnxrt_qlinearops", "preprocess")
        cls.pt_img = Image.fromarray(cls.img.astype(np.uint8))
        cls.tf_img = tf.constant(cls.img)
        _ = TRANSFORMS("tensorflow", "postprocess")
        _ = TRANSFORMS("pytorch", "postprocess")
        _ = TRANSFORMS("onnxrt_qlinearops", "postprocess")
        _ = TRANSFORMS("onnxrt_integerops", "postprocess")

    def testCast(self):
        args = {"dtype": "int64"}
        tf_func = TestSameTransfoms.tf_trans["Cast"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result[0][0][0].dtype, "int64")
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result[0][0][0].dtype, "int64")
        ox_func = TestSameTransfoms.ox_trans["Cast"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))
        self.assertEqual(ox_result[0][0][0].dtype, "int64")

        totensor = TestSameTransfoms.pt_trans["ToTensor"]()
        cast = TestSameTransfoms.pt_trans["Cast"](**args)
        pt_func = TestSameTransfoms.pt_trans["Compose"]([totensor, cast])
        pt_result = pt_func((TestSameTransfoms.pt_img, None))
        self.assertEqual(pt_result[0][0][0].dtype, torch.int64)

    def testCropToBoundingBox(self):
        args = {"offset_height": 2, "offset_width": 2, "target_height": 5, "target_width": 5}
        pt_func = TestSameTransfoms.pt_trans["CropToBoundingBox"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(pt_result.size, (5, 5))

        ox_func = TestSameTransfoms.ox_trans["CropToBoundingBox"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(ox_result.shape, (5, 5, 3))

        tf_func = TestSameTransfoms.tf_trans["CropToBoundingBox"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (5, 5, 3))

    def testNormalize(self):
        args = {}
        normalize = TestSameTransfoms.pt_trans["Normalize"](**args)
        totensor = TestSameTransfoms.pt_trans["ToTensor"]()
        pt_func = TestSameTransfoms.pt_trans["Compose"]([totensor, normalize])
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(TestSameTransfoms.img.astype(np.uint8)[0][0][0] / 255.0, pt_result[0][0][0])
        args = {"std": [0.0]}
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans["Normalize"](**args)

    def testRescale(self):
        ox_func = TestSameTransfoms.ox_trans["Rescale"]()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertAlmostEqual(ox_result[1][2][0], TestSameTransfoms.img[1][2][0] / 255.0)

    def testTranspose(self):
        args = {"perm": [2, 0, 1]}
        tf_func = TestSameTransfoms.tf_trans["Transpose"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["Transpose"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_transpose = TestSameTransfoms.pt_trans["Transpose"](**args)
        pt_totensor = TestSameTransfoms.pt_trans["ToTensor"]()
        pt_compose = TestSameTransfoms.pt_trans["Compose"]([pt_totensor, pt_transpose])
        pt_result = pt_compose((TestSameTransfoms.pt_img, None))[0]

        self.assertEqual(tf_result.shape, (3, 10, 10))
        self.assertEqual(ox_result.shape, (3, 10, 10))
        self.assertEqual(pt_result.shape, (10, 3, 10))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (3, 10, 10))

    def testCenterCrop(self):
        args = {"size": [4, 4]}
        tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["CenterCrop"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))
        self.assertEqual(pt_result.size, (4, 4))
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4, 4, 3))

        tf_result = tf_func((tf.constant(TestSameTransfoms.img.reshape((1, 10, 10, 3))), None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (1, 4, 4, 3))

        args = {"size": 4}
        tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["CenterCrop"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))
        self.assertEqual(pt_result.size, (4, 4))
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))

        args = {"size": [4]}
        tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))

        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
            tf_result = tf_func((np.array([[TestSameTransfoms.img]]), None))
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
            tf_result = tf_func((tf.constant(TestSameTransfoms.img.reshape((1, 1, 10, 10, 3))), None))

        args = {"size": [20]}
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
            tf_result = tf_func((TestSameTransfoms.img, None))
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans["CenterCrop"](**args)
            tf_result = tf_func((TestSameTransfoms.tf_img, None))

    def testResizeWithRatio(self):
        args = {"padding": True}
        label = [[0.1, 0.1, 0.5, 0.5], [], [], []]
        tf_func = TestSameTransfoms.tf_trans["ResizeWithRatio"](**args)
        tf_result = tf_func((TestSameTransfoms.img, label))[0]
        self.assertEqual(tf_result.shape, (1365, 1365, 3))

        args = {"padding": False}
        tf_func = TestSameTransfoms.tf_trans["ResizeWithRatio"](**args)
        tf_result = tf_func((TestSameTransfoms.img, label))[0]
        self.assertTrue((tf_result.shape[0] == 800 or tf_result.shape[1] == 1365))

    def testResize(self):
        tf_func = TestSameTransfoms.tf_trans["Resize"](**{"size": [4, 5]})
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["Resize"](**{"size": [4, 5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (5, 4, 3))
        self.assertEqual(pt_result.size, (5, 4))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4, 5, 3))

        args = {"size": 4}
        tf_func = TestSameTransfoms.tf_trans["Resize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["Resize"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))
        self.assertEqual(pt_result.size, (4, 4))

        args = {"size": [4]}
        tf_func = TestSameTransfoms.tf_trans["Resize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))

        args = {"size": 4, "interpolation": "test"}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans["Resize"](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans["Resize"](**args)

    def testRandomResizedCrop(self):
        tf_func = TestSameTransfoms.tf_trans["RandomResizedCrop"](**{"size": [4, 5]})
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["RandomResizedCrop"](**{"size": [4, 5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (5, 4, 3))
        self.assertEqual(pt_result.size, (5, 4))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4, 5, 3))

        args = {"size": [4]}
        tf_func = TestSameTransfoms.tf_trans["RandomResizedCrop"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))

        args = {"size": 4}
        tf_func = TestSameTransfoms.tf_trans["RandomResizedCrop"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["RandomResizedCrop"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (4, 4, 3))
        self.assertEqual(pt_result.size, (4, 4))

        args = {"size": 4, "scale": (0.8, 0.2)}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans["RandomResizedCrop"](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans["RandomResizedCrop"](**args)

        args = {"size": 4, "interpolation": "test"}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans["RandomResizedCrop"](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans["RandomResizedCrop"](**args)

    def testCropResize(self):
        args = {"x": 0, "y": 0, "width": 10, "height": 10, "size": [5, 5]}
        tf_func = TestSameTransfoms.tf_trans["CropResize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["CropResize"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["CropResize"](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))
        self.assertEqual(ox_result.shape, (5, 5, 3))
        self.assertEqual(pt_result.size, (5, 5))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (5, 5, 3))

        args = {"x": 0, "y": 0, "width": 10, "height": 10, "size": 5}
        tf_func = TestSameTransfoms.tf_trans["CropResize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["CropResize"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))
        self.assertEqual(ox_result.shape, (5, 5, 3))

        args = {"x": 0, "y": 0, "width": 10, "height": 10, "size": [5]}
        tf_func = TestSameTransfoms.tf_trans["CropResize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["CropResize"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))
        self.assertEqual(ox_result.shape, (5, 5, 3))

        args = {"x": 0, "y": 0, "width": 10, "height": 10, "size": [5, 5]}
        tf_func = TestSameTransfoms.tf_trans["CropResize"](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["CropResize"](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))
        self.assertEqual(ox_result.shape, (5, 5, 3))

        args = {"x": 0, "y": 0, "width": 10, "height": 10, "size": 5, "interpolation": "test"}
        with self.assertRaises(ValueError):
            TestSameTransfoms.ox_trans["CropResize"](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans["CropResize"](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans["CropResize"](**args)

    def testRandomHorizontalFlip(self):
        tf_func = TestSameTransfoms.tf_trans["RandomHorizontalFlip"]()
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["RandomHorizontalFlip"]()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["RandomHorizontalFlip"]()
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertTrue(
            (np.array(TestSameTransfoms.pt_img) == np.array(pt_result)).all()
            or (np.fliplr(np.array(TestSameTransfoms.pt_img)) == np.array(pt_result)).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or (np.fliplr(TestSameTransfoms.img) == tf_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == ox_result).all() or (np.fliplr(TestSameTransfoms.img) == ox_result).all()
        )

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or (np.fliplr(TestSameTransfoms.img) == tf_result).all()
        )

    def testRandomVerticalFlip(self):
        tf_func = TestSameTransfoms.tf_trans["RandomVerticalFlip"]()
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans["RandomVerticalFlip"]()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans["RandomVerticalFlip"]()
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertTrue(
            (np.array(TestSameTransfoms.pt_img) == np.array(pt_result)).all()
            or (np.flipud(np.array(TestSameTransfoms.pt_img)) == np.array(pt_result)).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or (np.flipud(TestSameTransfoms.img) == tf_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == ox_result).all() or (np.flipud(TestSameTransfoms.img) == ox_result).all()
        )
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or (np.flipud(TestSameTransfoms.img) == tf_result).all()
        )


class TestTFTransorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.ones([10, 10, 3])
        cls.tf_img = tf.constant(cls.img)
        cls.transforms = TRANSFORMS("tensorflow", "preprocess")
        cls.tf_img = tf.constant(cls.img)

    def testRandomCrop(self):
        args = {"size": [50]}
        transform = TestTFTransorm.transforms["RandomCrop"](**args)
        self.assertRaises(ValueError, transform, (TestTFTransorm.img, None))
        self.assertRaises(ValueError, transform, (TestTFTransorm.tf_img, None))

        args = {"size": [5, 5]}
        transform = TestTFTransorm.transforms["RandomCrop"](**args)
        img_result = transform((TestTFTransorm.img, None))[0]
        self.assertEqual(img_result.shape, (5, 5, 3))
        tf_result = transform((tf.constant(TestTFTransorm.img.reshape((1, 10, 10, 3))), None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (1, 5, 5, 3))

        args = {"size": [10, 10]}
        transform = TestTFTransorm.transforms["RandomCrop"](**args)
        img_result = transform((TestTFTransorm.img, None))[0]
        self.assertEqual(img_result.shape, (10, 10, 3))
        tf_result = transform((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (10, 10, 3))

    def testPaddedCenterCrop(self):
        args = {"size": [4, 4]}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10, 10, 3))

        args = {"size": [4, 4], "crop_padding": 4}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))

        args = {"size": 4}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10, 10, 3))

        args = {"size": 4, "crop_padding": 4}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))

        args = {"size": [4]}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10, 10, 3))

        args = {"size": [4], "crop_padding": 4}
        tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5, 5, 3))

        args = {"size": [4, 5], "crop_padding": 4}
        with self.assertRaises(ValueError):
            tf_func = TestTFTransorm.transforms["PaddedCenterCrop"](**args)
            tf_result = tf_func((TestTFTransorm.img, None))

    def testRescale(self):
        transform = TestTFTransorm.transforms["Rescale"]()
        img_result = transform((TestTFTransorm.img, None))[0]
        comp_result = np.array(TestTFTransorm.img) / 255.0
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)

        tf_result = transform((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertAlmostEqual(tf_result[0][0][0], comp_result[0][0][0], places=5)

    def testNormalize(self):
        args = {"mean": [0.0, 0.0, 0.0], "std": [0.2, 0.5, 0.1]}
        normalize = TestTFTransorm.transforms["Normalize"](**args)
        img_result = normalize((TestTFTransorm.img, None))[0]
        comp_result = np.array(TestTFTransorm.img) / [0.2, 0.5, 0.1]
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)
        self.assertAlmostEqual(img_result[0][0][1], comp_result[0][0][1], places=5)
        self.assertAlmostEqual(img_result[0][0][2], comp_result[0][0][2], places=5)

        tf_result = normalize((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertAlmostEqual(tf_result[0][0][0], comp_result[0][0][0], places=5)

        args = {"mean": [0.0, 0.0, 0.0], "std": [0, 0, 0]}
        with self.assertRaises(ValueError):
            TestTFTransorm.transforms["Normalize"](**args)

    def testRandomResizedCrop(self):
        args = {"size": [50]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms["Compose"]([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        self.assertEqual(image_result.shape, (50, 50, 3))
        args = {"size": [100, 100]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms["Compose"]([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        self.assertEqual(image_result.shape, (100, 100, 3))
        tf_result = randomresizedcrop((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (100, 100, 3))
        args = {"size": [100, 100], "scale": (0.8, 0.1)}
        with self.assertRaises(ValueError):
            TestTFTransorm.transforms["RandomResizedCrop"](**args)

    def testSquadV1(self):
        import json
        import ssl
        import urllib

        ssl._create_default_https_context = ssl._create_unverified_context

        vocab_url = (
            "https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-large-uncased-vocab.txt"
        )
        urllib.request.urlretrieve(vocab_url, "./vocab.txt")
        label = [
            {
                "paragraphs": [
                    {
                        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.",
                        "qas": [
                            {
                                "answers": [
                                    {"answer_start": 177, "text": "Denver Broncos"},
                                    {"answer_start": 177, "text": "Denver Broncos"},
                                    {"answer_start": 177, "text": "Denver Broncos"},
                                ],
                                "question": "Which NFL team represented the AFC at Super Bowl 50?",
                                "id": "56be4db0acb8001400a502ec",
                            }
                        ],
                    }
                ]
            }
        ]
        fake_json = json.dumps({"data": label})
        with open("dev.json", "w") as f:
            f.write(fake_json)
        args = {"label_file": "./dev.json", "vocab_file": "./vocab.txt"}
        post_transforms = TRANSFORMS("tensorflow", "postprocess")
        squadv1 = post_transforms["SquadV1"](**args)

        preds_0 = np.array([1000000000])
        preds_1 = np.random.uniform(low=-12.3, high=6.8, size=(1, 384))
        preds_2 = np.random.uniform(low=-10.8, high=7.4, size=(1, 384))
        preds = [preds_0, preds_1, preds_2]
        result = squadv1((preds, label))
        self.assertTrue(result[1][0]["paragraphs"][0]["qas"][0]["id"] in result[0])
        os.remove("dev.json")
        os.remove("vocab.txt")


class TestAlignImageChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1 = np.random.random_sample([100, 100, 3]) * 255
        cls.img2 = np.random.random_sample([100, 100]) * 255
        cls.img3 = np.random.random_sample([100, 100, 4]) * 255
        cls.pt_img1 = Image.fromarray(cls.img1.astype(np.uint8))
        cls.pt_img2 = Image.fromarray(cls.img2.astype(np.uint8))
        cls.pt_img3 = Image.fromarray(cls.img3.astype(np.uint8))

    def testTensorflow(self):
        transforms = TRANSFORMS("tensorflow", "preprocess")
        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.img1.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.img2.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms["AlignImageChannel"](**{"dim": 3})
        image, _ = align((TestAlignImageChannel.img3.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 3)

        align = transforms["AlignImageChannel"](**{"dim": 2})
        self.assertRaises(ValueError, align, (TestAlignImageChannel.img1.astype(np.uint8), None))

        with self.assertRaises(ValueError):
            transforms["AlignImageChannel"](**{"dim": 5})

    def testONNX(self):
        transforms = TRANSFORMS("onnxrt_qlinearops", "preprocess")
        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.img1.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.img2.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms["AlignImageChannel"](**{"dim": 3})
        image, _ = align((TestAlignImageChannel.img3.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 3)

        align = transforms["AlignImageChannel"](**{"dim": 2})
        self.assertRaises(ValueError, align, (TestAlignImageChannel.img1.astype(np.uint8), None))

        with self.assertRaises(ValueError):
            transforms["AlignImageChannel"](**{"dim": 5})

    def testPyTorch(self):
        transforms = TRANSFORMS("pytorch", "preprocess")
        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.pt_img1, None))
        self.assertEqual(image.mode, "L")

        align = transforms["AlignImageChannel"](**{"dim": 1})
        image, _ = align((TestAlignImageChannel.pt_img2, None))
        self.assertEqual(image.mode, "L")

        align = transforms["AlignImageChannel"](**{"dim": 3})
        image, _ = align((TestAlignImageChannel.pt_img3, None))
        self.assertEqual(image.mode, "RGB")

        with self.assertRaises(ValueError):
            align = transforms["AlignImageChannel"](**{"dim": 2})

        with self.assertRaises(ValueError):
            transforms["AlignImageChannel"](**{"dim": 5})


class TestToArray(unittest.TestCase):
    def testParse(self):
        random_array = np.random.random_sample([10, 10, 3]) * 255
        random_array = random_array.astype(np.uint8)
        img1 = Image.fromarray(random_array)
        onnx_transforms = TRANSFORMS("onnxrt_qlinearops", "preprocess")
        onnx_parse = onnx_transforms["ToArray"]()
        img, _ = onnx_parse((img1, None))
        self.assertTrue(isinstance(img, np.ndarray))


class TestONNXTransfrom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([100, 100, 3]) * 255
        cls.transforms = TRANSFORMS("onnxrt_qlinearops", "preprocess")

    def testResize(self):
        args = {"size": [224]}
        resize = TestONNXTransfrom.transforms["Resize"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224, 224, 3))
        args = {"size": [100, 100], "interpolation": "test"}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["Resize"](**args)

        args = {"size": 224}
        resize = TestONNXTransfrom.transforms["Resize"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224, 224, 3))

        args = {"size": [224, 224]}
        resize = TestONNXTransfrom.transforms["Resize"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224, 224, 3))

    def testNormalize(self):
        args = {"mean": [0.0, 0.0, 0.0], "std": [0.29, 0.24, 0.25]}
        normalize = TestONNXTransfrom.transforms["Normalize"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([normalize])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertTrue((image_result[0] == np.array(TestONNXTransfrom.img) / [0.29, 0.24, 0.25]).all())

        args = {"mean": [0.0, 0.0, 0.0], "std": [0, 0, 0]}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["Normalize"](**args)

    def testRandomCrop(self):
        args = {"size": [50]}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (50, 50, 3))
        args = {"size": [1000, 1000]}
        with self.assertRaises(ValueError):
            trans = TestONNXTransfrom.transforms["RandomCrop"](**args)
            trans((TestONNXTransfrom.img, None))

        args = {"size": 50}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (50, 50, 3))

        args = {"size": [100, 100]}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (100, 100, 3))

    def testCenterCrop(self):
        args = {"size": [100]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([centercrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (100, 100, 3))
        args = {"size": 5}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        image_result = centercrop((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (5, 5, 3))
        args = {"size": [5, 6]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        image_result = centercrop((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (5, 6, 3))
        args = {"size": [150]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        with self.assertRaises(ValueError):
            centercrop((TestONNXTransfrom.img, None))

    def testRandomResizedCrop(self):
        args = {"size": [150]}
        randomresizedcrop = TestONNXTransfrom.transforms["RandomResizedCrop"](**args)
        compose = TestONNXTransfrom.transforms["Compose"]([randomresizedcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (150, 150, 3))
        args = {"size": [150, 150], "scale": (0.9, 0.3)}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["RandomResizedCrop"](**args)

        args = {"size": 150, "interpolation": "test"}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["RandomResizedCrop"](**args)


if __name__ == "__main__":
    unittest.main()
