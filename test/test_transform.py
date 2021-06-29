"""Tests for the transform module."""
import numpy as np
import random
import unittest
import os
from lpot.data import TRANSFORMS, DATALOADERS
from lpot.utils.create_obj_from_config import get_postprocess, create_dataset
from lpot.utils.utility import LazyImport
from PIL import Image
mx = LazyImport('mxnet')
tf = LazyImport('tensorflow')
torch = LazyImport('torch')
torchvision = LazyImport('torchvision')

random.seed(1)
np.random.seed(1)

class TestMetrics(unittest.TestCase):
    def test_tensorflow_2(self):
        image = np.ones([256, 256, 1])
        resize_kwargs = {"size":[224, 224]}
        transforms = TRANSFORMS(framework="tensorflow", process="preprocess")
        resize = transforms['Resize'](**resize_kwargs)
        random_crop_kwargs = {"size": 128}
        random_crop = transforms['RandomCrop'](**random_crop_kwargs)
        transform_list = [resize, random_crop]
        compose = transforms['Compose'](transform_list)
        image_result = compose((image, None))
        self.assertEqual(image_result[0].shape, (128, 128))

class TestONNXQLImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([600,600])*255

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('onnxrt_qlinearops', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224, random_crop=True)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)

class TestONNXITImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([600,600,3])*255

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('onnxrt_integerops', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)

    def testResizeWithAspectRatio(self):
        transforms = TRANSFORMS('onnxrt_integerops', "preprocess")
        transform = transforms['ResizeWithAspectRatio'](height=224, width=224)
        sample = (self.img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 256)
        self.assertEqual(len(resized_input[0]), 256)
        self.assertEqual(len(resized_input[0][0]), 3)

class TestTensorflowImagenetTransform(unittest.TestCase):
    tf.compat.v1.disable_v2_behavior()
    def testBilinearImagenetTransform(self):
        transforms = TRANSFORMS('tensorflow', "preprocess")
        transform = transforms['BilinearImagenet'](height=224, width=224)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 0)
        result = transform(sample)
        resized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)

        transforms = TRANSFORMS('onnxrt_qlinearops', "preprocess")
        transform = transforms['BilinearImagenet'](height=224, width=224)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 0)
        result = transform(sample)
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)
    
    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('tensorflow', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224, random_crop=True,
            random_flip_left_right=True)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 0)
        result = transform(sample)
        resized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(len(resized_input), 224)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 3)
    
    def testLabelShift(self):
        transforms = TRANSFORMS('tensorflow', "postprocess")
        transform = transforms['LabelShift'](label_shift=1)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 1001)
        label = transform(sample)[1]
        self.assertEqual(label, 1000)
    
    def testQuantizedInput(self):
        transforms = TRANSFORMS('tensorflow', "preprocess")
        transform = transforms['QuantizedInput'](dtype='uint8', scale=100)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 1001)
        result = transform(sample)
        quantized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertLessEqual(quantized_input.max(), 255)
        self.assertGreaterEqual(quantized_input.min(), 0)

        transform = transforms['QuantizedInput'](dtype='uint8')
        sample = (rand_input, 1001)
        result = transform(sample)
        quantized_input = result[0]
        self.assertLessEqual(quantized_input.max(), 1)
        self.assertGreaterEqual(quantized_input.min(), 0)

class TestDataConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([10,10,3])*255
        cls.mx_trans = TRANSFORMS('mxnet', 'preprocess')
        cls.pt_trans = TRANSFORMS('pytorch', 'preprocess')
 
    def testToPILImage(self):
        trans = TestDataConversion.pt_trans['ToPILImage']()
        image, _ = trans((TestDataConversion.img.astype(np.uint8), None))
        self.assertTrue(isinstance(image, Image.Image))

    def testToTensor(self):
        trans = TestDataConversion.pt_trans['ToTensor']()
        image, _ = trans((TestDataConversion.img.astype(np.uint8), None))
        self.assertTrue(isinstance(image, torch.Tensor))

        trans = TestDataConversion.mx_trans['ToTensor']()
        image, _ = trans((mx.nd.array(TestDataConversion.img), None))
        self.assertTrue(isinstance(image, mx.ndarray.NDArray)) # pylint: disable=no-member

    def testToNDArray(self):
        trans = TestDataConversion.mx_trans['ToNDArray']()
        image, _ = trans((TestDataConversion.img.astype(np.uint8), None))
        self.assertTrue(isinstance(image, mx.ndarray.NDArray))

class TestSameTransfoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([10,10,3])*255
        cls.tf_trans = TRANSFORMS('tensorflow', 'preprocess')
        cls.pt_trans = TRANSFORMS('pytorch', 'preprocess')
        cls.mx_trans = TRANSFORMS('mxnet', 'preprocess')
        cls.ox_trans = TRANSFORMS('onnxrt_qlinearops', 'preprocess')
        cls.mx_img = mx.nd.array(cls.img.astype(np.uint8))
        cls.pt_img = Image.fromarray(cls.img.astype(np.uint8))
        cls.tf_img = tf.constant(cls.img)
        _ = TRANSFORMS('tensorflow', 'postprocess')
        _ = TRANSFORMS('pytorch', 'postprocess')
        _ = TRANSFORMS('mxnet', 'postprocess')
        _ = TRANSFORMS('onnxrt_qlinearops' , 'postprocess')
        _ = TRANSFORMS('onnxrt_integerops', 'postprocess')

    def testCast(self):
        args = {'dtype': 'int64'}
        tf_func = TestSameTransfoms.tf_trans['Cast'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result[0][0][0].dtype, 'int64')
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result[0][0][0].dtype, 'int64')
        mx_func = TestSameTransfoms.mx_trans['Cast'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        self.assertEqual(mx_result[0][0][0].dtype, np.int64)
        ox_func = TestSameTransfoms.ox_trans['Cast'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))
        self.assertEqual(ox_result[0][0][0].dtype, 'int64')

        totensor = TestSameTransfoms.pt_trans['ToTensor']()
        cast = TestSameTransfoms.pt_trans['Cast'](**args)
        pt_func = TestSameTransfoms.pt_trans['Compose']([totensor, cast])
        pt_result = pt_func((TestSameTransfoms.pt_img, None))
        self.assertEqual(pt_result[0][0][0].dtype, torch.int64)

    def testCropToBoundingBox(self):
        args = {'offset_height':2, 'offset_width':2, 'target_height':5, 'target_width':5}
        pt_func = TestSameTransfoms.pt_trans['CropToBoundingBox'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(pt_result.size, (5,5))

        ox_func = TestSameTransfoms.ox_trans['CropToBoundingBox'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(ox_result.shape, (5,5,3))

        mx_func = TestSameTransfoms.mx_trans['CropToBoundingBox'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(mx_result.shape, (5,5,3))

        tf_func = TestSameTransfoms.tf_trans['CropToBoundingBox'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (5,5,3))
 
    def testNormalize(self):
        args = {}
        normalize = TestSameTransfoms.pt_trans['Normalize'](**args)
        totensor = TestSameTransfoms.pt_trans['ToTensor']()
        pt_func = TestSameTransfoms.pt_trans['Compose']([totensor, normalize])
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(TestSameTransfoms.img.astype(
                np.uint8)[0][0][0]/255., pt_result[0][0][0])
        args = {'std': [0.]}
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans['Normalize'](**args)

    def testRescale(self):
        ox_func = TestSameTransfoms.ox_trans['Rescale']()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertAlmostEqual(ox_result[1][2][0], TestSameTransfoms.img[1][2][0]/255.)

    def testTranspose(self):
        args = {'perm': [2, 0, 1]}
        tf_func = TestSameTransfoms.tf_trans['Transpose'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['Transpose'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Transpose'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        pt_transpose = TestSameTransfoms.pt_trans['Transpose'](**args)
        pt_totensor = TestSameTransfoms.pt_trans['ToTensor']()
        pt_compose = TestSameTransfoms.pt_trans['Compose']([pt_totensor, pt_transpose])
        pt_result = pt_compose((TestSameTransfoms.pt_img, None))[0]
 
        self.assertEqual(tf_result.shape, (3,10,10))
        self.assertEqual(ox_result.shape, (3,10,10))
        self.assertEqual(mx_result.shape, (3,10,10))
        self.assertEqual(pt_result.shape, (10,3,10))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (3,10,10))
 
    def testCenterCrop(self):
        args = {'size':[4,4]}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['CenterCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CenterCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))
        self.assertEqual(np.array(pt_result)[0][0][0], mx_result.asnumpy()[0][0][0])
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4,4,3))

        tf_result = tf_func((tf.constant(TestSameTransfoms.img.reshape((1,10,10,3))), None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (1,4,4,3))

        args = {'size':4}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['CenterCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CenterCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))
        self.assertEqual(np.array(pt_result)[0][0][0], mx_result.asnumpy()[0][0][0])
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))
        
        args = {'size':[4]}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))

        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
            tf_result = tf_func((np.array([[TestSameTransfoms.img]]), None))
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
            tf_result = tf_func((tf.constant(TestSameTransfoms.img.reshape((1,1,10,10,3))), None))

        args = {'size':[20]}
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
            tf_result = tf_func((TestSameTransfoms.img, None))
        with self.assertRaises(ValueError):
            tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
            tf_result = tf_func((TestSameTransfoms.tf_img, None))

    def testResizeWithRatio(self):
        args = {'padding': True}
        label = [[0.1,0.1,0.5,0.5], [], [], []]
        tf_func = TestSameTransfoms.tf_trans['ResizeWithRatio'](**args)
        tf_result = tf_func((TestSameTransfoms.img, label))[0]
        self.assertEqual(tf_result.shape, (1365,1365,3))
 
        args = {'padding': False}
        tf_func = TestSameTransfoms.tf_trans['ResizeWithRatio'](**args)
        tf_result = tf_func((TestSameTransfoms.img, label))[0]
        self.assertTrue((tf_result.shape[0]==800 or tf_result.shape[1] ==1365))
 
    def testResize(self):
        tf_func = TestSameTransfoms.tf_trans['Resize'](**{'size':[4,5]})
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['Resize'](**{'size':[4,5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Resize'](**{'size':[4,5]})
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (5,4,3))
        self.assertEqual(pt_result.size, (5,4))
        self.assertEqual(mx_result.shape, (4,5,3))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4,5,3))

        args = {'size': 4}
        tf_func = TestSameTransfoms.tf_trans['Resize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['Resize'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Resize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))

        args = {'size': [4]}
        tf_func = TestSameTransfoms.tf_trans['Resize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Resize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(mx_result.shape, (4,4,3))

        args = {'size': 4, 'interpolation':'test'}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans['Resize'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans['Resize'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.mx_trans['Resize'](**args)
 
    def testRandomResizedCrop(self):
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**{'size':[4,5]})
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['RandomResizedCrop'](**{'size':[4,5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomResizedCrop'](**{'size':[4,5]})
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (5,4,3))
        self.assertEqual(pt_result.size, (5,4))
        self.assertEqual(mx_result.shape, (4,5,3))
    
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4,5,3))

        args = {'size': [4]}
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        mx_func = TestSameTransfoms.mx_trans['RandomResizedCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(mx_result.shape, (4,4,3))

        args = {'size': 4}
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['RandomResizedCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomResizedCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))

        args = {'size': 4, 'scale':(0.8, 0.2)}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans['RandomResizedCrop'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.mx_trans['RandomResizedCrop'](**args)
        
        args = {'size': 4, 'interpolation':'test'}
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans['RandomResizedCrop'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.mx_trans['RandomResizedCrop'](**args)

    def testCropResize(self):
        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':[5,5]}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['CropResize'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))
        self.assertEqual(pt_result.size, (5,5))

        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (5,5,3))

        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':5}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))

        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':[5]}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))

        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':[5,5]}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))

        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':5, 'interpolation':'test'}
        with self.assertRaises(ValueError):
            TestSameTransfoms.ox_trans['CropResize'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.mx_trans['CropResize'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.tf_trans['CropResize'](**args)
        with self.assertRaises(ValueError):
            TestSameTransfoms.pt_trans['CropResize'](**args)

    def testRandomHorizontalFlip(self):
        tf_func = TestSameTransfoms.tf_trans['RandomHorizontalFlip']()
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['RandomHorizontalFlip']()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['RandomHorizontalFlip']()
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomHorizontalFlip']()
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertTrue(
            (np.array(TestSameTransfoms.pt_img) == np.array(pt_result)).all() or
            (np.fliplr(np.array(TestSameTransfoms.pt_img)) == np.array(pt_result)).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or
            (np.fliplr(TestSameTransfoms.img) == tf_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == ox_result).all() or
            (np.fliplr(TestSameTransfoms.img) == ox_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.mx_img.asnumpy() == mx_result.asnumpy()).all() or
            (np.fliplr(TestSameTransfoms.mx_img.asnumpy()) == mx_result.asnumpy()).all()
        )
    
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or
            (np.fliplr(TestSameTransfoms.img) == tf_result).all()
        )

    def testRandomVerticalFlip(self):
        tf_func = TestSameTransfoms.tf_trans['RandomVerticalFlip']()
        tf_result = tf_func((TestSameTransfoms.img, None))[0]
        ox_func = TestSameTransfoms.ox_trans['RandomVerticalFlip']()
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        pt_func = TestSameTransfoms.pt_trans['RandomVerticalFlip']()
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomVerticalFlip']()
        mx_result = mx_func((TestSameTransfoms.mx_img, None))[0]
        self.assertTrue(
            (np.array(TestSameTransfoms.pt_img) == np.array(pt_result)).all() or
            (np.flipud(np.array(TestSameTransfoms.pt_img)) == np.array(pt_result)).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or
            (np.flipud(TestSameTransfoms.img) == tf_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.img == ox_result).all() or
            (np.flipud(TestSameTransfoms.img) == ox_result).all()
        )
        self.assertTrue(
            (TestSameTransfoms.mx_img.asnumpy() == mx_result.asnumpy()).all() or
            (np.flipud(TestSameTransfoms.mx_img.asnumpy()) == mx_result.asnumpy()).all()
        )
 
        tf_result = tf_func((TestSameTransfoms.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertTrue(
            (TestSameTransfoms.img == tf_result).all() or
            (np.flipud(TestSameTransfoms.img) == tf_result).all()
        )

class TestTFTransorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.ones([10,10,3])
        cls.tf_img = tf.constant(cls.img)
        cls.transforms = TRANSFORMS('tensorflow', 'preprocess')
        cls.tf_img = tf.constant(cls.img)

    def testRandomCrop(self):
        args = {'size': [50]}
        transform = TestTFTransorm.transforms['RandomCrop'](**args)
        self.assertRaises(ValueError, transform, (TestTFTransorm.img, None))
        self.assertRaises(ValueError, transform, (TestTFTransorm.tf_img, None))
 
        args = {'size': [5, 5]}
        transform = TestTFTransorm.transforms['RandomCrop'](**args)
        img_result = transform((TestTFTransorm.img, None))[0]
        self.assertEqual(img_result.shape, (5,5,3))
        tf_result = transform((tf.constant(TestTFTransorm.img.reshape((1,10,10,3))), None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (1,5,5,3))

        args = {'size': [10,10]}
        transform = TestTFTransorm.transforms['RandomCrop'](**args)
        img_result = transform((TestTFTransorm.img, None))[0]
        self.assertEqual(img_result.shape, (10,10,3))
        tf_result = transform((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (10,10,3))

    def testPaddedCenterCrop(self):
        args = {'size':[4,4]}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10,10,3))

        args = {'size':[4,4], 'crop_padding': 4}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))

        args = {'size':4}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10,10,3))

        args = {'size':4, 'crop_padding':4}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))

        args = {'size':[4]}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (10,10,3))
        
        args = {'size':[4], 'crop_padding':4}
        tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
        tf_result = tf_func((TestTFTransorm.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))

        args = {'size':[4,5], 'crop_padding':4}
        with self.assertRaises(ValueError):
            tf_func = TestTFTransorm.transforms['PaddedCenterCrop'](**args)
            tf_result = tf_func((TestTFTransorm.img, None))

    def testRescale(self):
        transform = TestTFTransorm.transforms['Rescale']()
        img_result = transform((TestTFTransorm.img, None))[0]
        comp_result = np.array(TestTFTransorm.img)/255.
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)

        tf_result = transform((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertAlmostEqual(tf_result[0][0][0], comp_result[0][0][0], places=5) 

    def testNormalize(self):
        args = {'mean':[0.0,0.0,0.0], 'std':[0.2, 0.5, 0.1]}
        normalize = TestTFTransorm.transforms['Normalize'](**args)
        img_result = normalize((TestTFTransorm.img, None))[0]
        comp_result = np.array(TestTFTransorm.img)/[0.2, 0.5, 0.1]
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)
        self.assertAlmostEqual(img_result[0][0][1], comp_result[0][0][1], places=5)
        self.assertAlmostEqual(img_result[0][0][2], comp_result[0][0][2], places=5)
        
        tf_result = normalize((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertAlmostEqual(tf_result[0][0][0], comp_result[0][0][0], places=5)

        args = {'mean':[0.0,0.0,0.0], 'std':[0, 0, 0]}
        with self.assertRaises(ValueError):
            TestTFTransorm.transforms["Normalize"](**args)

    def testRandomResizedCrop(self):
        args = {'size':[50]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        self.assertEqual(image_result.shape, (50,50,3))
        args = {'size':[100, 100]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        self.assertEqual(image_result.shape, (100,100,3))
        tf_result = randomresizedcrop((TestTFTransorm.tf_img, None))[0]
        tf_result = tf_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (100,100,3)) 
        args = {'size':[100, 100], 'scale':(0.8, 0.1)}
        with self.assertRaises(ValueError):
            TestTFTransorm.transforms["RandomResizedCrop"](**args)

    def testSquadV1(self):
        import urllib
        import json
        vocab_url = "https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-large-uncased-vocab.txt"
        urllib.request.urlretrieve(vocab_url, "./vocab.txt")
        label = [{
            "paragraphs":[
                {'context': 
                    'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.',
                'qas': [{
                    'answers': [
                        {'answer_start': 177, 'text': 'Denver Broncos'}, 
                        {'answer_start': 177, 'text': 'Denver Broncos'}, 
                        {'answer_start': 177, 'text': 'Denver Broncos'}], 
                    'question': 'Which NFL team represented the AFC at Super Bowl 50?', 
                    'id': '56be4db0acb8001400a502ec'}]
                }
            ]
        }]
        fake_json = json.dumps({'data': label})
        with open('dev.json', 'w') as f:
            f.write(fake_json)
        args = {
            'label_file': './dev.json',
            'vocab_file': './vocab.txt'
        }
        post_transforms = TRANSFORMS('tensorflow', 'postprocess')
        squadv1 = post_transforms['SquadV1'](**args)
        
        preds_0 = np.array([1000000000])
        preds_1 = np.random.uniform(low=-12.3, high=6.8, size=(1,384))
        preds_2 = np.random.uniform(low=-10.8, high=7.4, size=(1,384))
        preds = [preds_0, preds_1, preds_2]
        result = squadv1((preds, label))
        self.assertTrue(result[1][0]['paragraphs'][0]['qas'][0]['id'] in result[0])
        os.remove('dev.json')
        os.remove('vocab.txt')
 
class TestAlignImageChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1 = np.random.random_sample([100,100,3]) * 255
        cls.img2 = np.random.random_sample([100,100]) * 255
        cls.img3 = np.random.random_sample([100,100,4]) * 255
        cls.pt_img1 = Image.fromarray(cls.img1.astype(np.uint8))
        cls.pt_img2 = Image.fromarray(cls.img2.astype(np.uint8))
        cls.pt_img3 = Image.fromarray(cls.img3.astype(np.uint8))
 
    def testTensorflow(self):
        transforms = TRANSFORMS('tensorflow', 'preprocess')
        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img1.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img2.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':3})
        image, _ = align((TestAlignImageChannel.img3.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 3)

        align = transforms['AlignImageChannel'](**{'dim':2})
        self.assertRaises(ValueError, align, 
                (TestAlignImageChannel.img1.astype(np.uint8), None))

        with self.assertRaises(ValueError):
            transforms['AlignImageChannel'](**{'dim':5})

    def testONNX(self):
        transforms = TRANSFORMS('onnxrt_qlinearops', 'preprocess')
        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img1.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img2.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':3})
        image, _ = align((TestAlignImageChannel.img3.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 3)

        align = transforms['AlignImageChannel'](**{'dim':2})
        self.assertRaises(ValueError, align, 
                (TestAlignImageChannel.img1.astype(np.uint8), None))

        with self.assertRaises(ValueError):
            transforms['AlignImageChannel'](**{'dim':5})

    def testPyTorch(self):
        transforms = TRANSFORMS('pytorch', 'preprocess')
        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.pt_img1, None))
        self.assertEqual(image.mode, 'L')

        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.pt_img2, None))
        self.assertEqual(image.mode, 'L')

        align = transforms['AlignImageChannel'](**{'dim':3})
        image, _ = align((TestAlignImageChannel.pt_img3, None))
        self.assertEqual(image.mode, 'RGB')

        with self.assertRaises(ValueError):
            align = transforms['AlignImageChannel'](**{'dim':2})

        with self.assertRaises(ValueError):
            transforms['AlignImageChannel'](**{'dim':5})

    def testMXNet(self):
        transforms = TRANSFORMS('mxnet', 'preprocess')
        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img1.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':1})
        image, _ = align((TestAlignImageChannel.img2.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 1)

        align = transforms['AlignImageChannel'](**{'dim':3})
        image, _ = align((TestAlignImageChannel.img3.astype(np.uint8), None))
        self.assertEqual(image.shape[-1], 3)

        align = transforms['AlignImageChannel'](**{'dim':2})
        self.assertRaises(ValueError, align, 
                (TestAlignImageChannel.img1.astype(np.uint8), None))

        with self.assertRaises(ValueError):
            transforms['AlignImageChannel'](**{'dim':5})

class TestToArray(unittest.TestCase):
    def testParse(self):
        random_array = np.random.random_sample([10,10,3]) * 255
        random_array = random_array.astype(np.uint8)
        img1 = Image.fromarray(random_array)
        onnx_transforms = TRANSFORMS('onnxrt_qlinearops', 'preprocess')
        onnx_parse = onnx_transforms['ToArray']()
        img, _ = onnx_parse((img1, None))
        self.assertTrue(isinstance(img, np.ndarray))

        mxnet_transforms = TRANSFORMS('mxnet', 'preprocess')
        mxnet_parse = mxnet_transforms['ToArray']()
        img, _ = mxnet_parse((mx.nd.array(random_array), None))
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertRaises(ValueError, mxnet_parse, ([1,2], None))

class TestMXNetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        array = np.random.random_sample([100,100,3]) * 255
        cls.img = mx.nd.array(array)
        cls.transforms = TRANSFORMS('mxnet', 'preprocess')

    def testRandomCrop(self):
        args = {'size':[50]}
        randomcrop = TestMXNetTransform.transforms["RandomCrop"](**args)
        compose = TestMXNetTransform.transforms['Compose']([randomcrop])
        image_result = compose((TestMXNetTransform.img, None))
        self.assertEqual(image_result[0].shape, (50,50,3))

    def testNormalize(self):
        args = {'mean':[0.0,0.0,0.0], 'std':[0.29, 0.24, 0.25]}
        normalize = TestMXNetTransform.transforms['Normalize'](**args)
        image_result = normalize((TestMXNetTransform.img, None))
        self.assertAlmostEqual(image_result[0].asnumpy()[0][0][0],
                (TestMXNetTransform.img.asnumpy()/[0.29])[0][0][0], places=3)

class TestONNXTransfrom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.random_sample([100,100,3]) * 255
        cls.transforms = TRANSFORMS('onnxrt_qlinearops', 'preprocess')

    def testResize(self):
        args = {'size':[224]}
        resize = TestONNXTransfrom.transforms['Resize'](**args)
        compose = TestONNXTransfrom.transforms['Compose']([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224,224,3))
        args = {'size':[100, 100], 'interpolation':'test'}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms['Resize'](**args)

        args = {'size':224}
        resize = TestONNXTransfrom.transforms['Resize'](**args)
        compose = TestONNXTransfrom.transforms['Compose']([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224,224,3))
        
        args = {'size':[224,224]}
        resize = TestONNXTransfrom.transforms['Resize'](**args)
        compose = TestONNXTransfrom.transforms['Compose']([resize])
        image_result = compose((self.img, None))
        self.assertEqual(image_result[0].shape, (224,224,3))
        
    def testNormalize(self):
        args = {'mean':[0.0,0.0,0.0], 'std':[0.29, 0.24, 0.25]}
        normalize = TestONNXTransfrom.transforms['Normalize'](**args)
        compose = TestONNXTransfrom.transforms['Compose']([normalize])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertTrue(
            (image_result[0] == np.array(TestONNXTransfrom.img)/[0.29, 0.24, 0.25]).all())

        args = {'mean':[0.0,0.0,0.0], 'std':[0,0,0]}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["Normalize"](**args)

    def testRandomCrop(self):
        args = {'size':[50]}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (50,50,3))
        args = {'size':[1000, 1000]}
        with self.assertRaises(ValueError):
            trans = TestONNXTransfrom.transforms["RandomCrop"](**args)
            trans((TestONNXTransfrom.img, None))

        args = {'size':50}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (50,50,3))
        
        args = {'size':[100,100]}
        randomcrop = TestONNXTransfrom.transforms["RandomCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([randomcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (100,100,3))
        
    def testCenterCrop(self):
        args = {'size':[100]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([centercrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (100,100,3))
        args = {'size': 5}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        image_result = centercrop((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (5,5,3))
        args = {'size': [5, 6]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        image_result = centercrop((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (5,6,3))
        args = {'size':[150]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        with self.assertRaises(ValueError):
            centercrop((TestONNXTransfrom.img, None))

    def testRandomResizedCrop(self):
        args = {'size':[150]}
        randomresizedcrop = TestONNXTransfrom.transforms["RandomResizedCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (150,150,3))
        args = {'size':[150, 150], 'scale':(0.9, 0.3)}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["RandomResizedCrop"](**args)

        args = {'size':150, 'interpolation':'test'}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["RandomResizedCrop"](**args)

class TestImagenetTransform(unittest.TestCase):
    def testParseDecodeImagenet(self):
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpeg')

        image = tf.compat.v1.gfile.FastGFile('test.jpeg','rb').read()
        label = 10
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image])),
            'image/class/label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
            'image/object/bbox/xmin': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10])),
            'image/object/bbox/ymin': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[20])),
            'image/object/bbox/xmax': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100])),
            'image/object/bbox/ymax': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[200])),
        }))
        with tf.io.TFRecordWriter('test-0-of-0') as writer:
            writer.write(example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow', {'ImageRecord':{'root':'./'}}, {'ParseDecodeImagenet':{}}, None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0][0], 10)
            break

        from lpot.experimental.data.transforms.imagenet_transform import ParseDecodeImagenet
        func = ParseDecodeImagenet()
        out = func(example.SerializeToString())
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (100,100,3))

        from lpot.experimental.data.datasets.dataset import TensorflowTFRecordDataset
        ds = TensorflowTFRecordDataset('test-0-of-0', func)
        dataloader = DATALOADERS['tensorflow'](dataset=ds, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0][0], 10)
            break

        os.remove('test-0-of-0')
        os.remove('test.jpeg')

class TestCOCOTransform(unittest.TestCase):
    def testCOCODecode(self):
        tf.compat.v1.disable_eager_execution() 

        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpeg')

        image = tf.compat.v1.gfile.FastGFile('test.jpeg','rb').read()
        source_id = '000000397133.jpg'.encode('utf-8')
        label = 'person'.encode('utf-8')
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image])),
            'image/object/class/text':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[label])),
            'image/source_id':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[source_id])),
            'image/object/bbox/xmin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10])),
            'image/object/bbox/ymin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10])),
            'image/object/bbox/xmax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100])),
            'image/object/bbox/ymax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100])),
        }))

        with tf.io.TFRecordWriter('test.record') as writer:
            writer.write(example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow', {'COCORecord':{'root':'test.record'}}, 
            {'ParseDecodeCoco':{}, 'Resize': {'size': 50}, 'Cast':{'dtype':'int64'},
            'CropToBoundingBox':{'offset_height':2, 'offset_width':2, 'target_height':5, 'target_width':5},
            'CenterCrop':{'size':[4,4]},
            'RandomResizedCrop':{'size':[4,5]},
            }, None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,4,5,3))
            self.assertEqual(labels[0].shape, (1,1,4))

        from lpot.experimental.data.transforms.transform import TensorflowResizeWithRatio
        from lpot.experimental.data.datasets.coco_dataset import ParseDecodeCoco
        func = ParseDecodeCoco()
        out = func(example.SerializeToString())
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (100,100,3))

        func = ParseDecodeCoco()
        out = func(example.SerializeToString())
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (100,100,3))

        func = TensorflowResizeWithRatio(**{'padding':True})
        out = func(out)
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (1365,1365,3))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image])),
            'image/source_id':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[source_id])),
            'image/object/bbox/xmin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10])),
            'image/object/bbox/ymin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10])),
            'image/object/bbox/xmax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100])),
            'image/object/bbox/ymax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100])),
        }))

        with tf.io.TFRecordWriter('test2.record') as writer:
            writer.write(example.SerializeToString())
        self.assertRaises(ValueError, create_dataset,
            'tensorflow', {'COCORecord':{'root':'test2.record'}}, None, None)

        os.remove('test2.record')
        os.remove('test.record')
        os.remove('test.jpeg')

class TestVOCTransform(unittest.TestCase):
    def testVOCDecode(self):
        import shutil
        tf.compat.v1.disable_eager_execution() 

        def _bytes_list_feature(values):
            import six
            def norm2bytes(value):
                return value.encode() if isinstance(value, str) and six.PY3 else value
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))

        def _int64_list_feature(values):
            import collections
            if not isinstance(values, collections.Iterable):
                values = [values]
            return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpg')
        random_array = np.random.random_sample([100,100,3]) * 0
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.png')
        image_data = tf.compat.v1.gfile.GFile('test.jpg', 'rb').read()
        seg_data = tf.compat.v1.gfile.GFile('test.png', 'rb').read()
        filename = 'test'

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_list_feature(image_data),
            'image/filename': _bytes_list_feature(filename),
            'image/format': _bytes_list_feature('png'),
            'image/height': _int64_list_feature(100),
            'image/width': _int64_list_feature(100),
            'image/channels': _int64_list_feature(3),
            'image/segmentation/class/encoded': (
                _bytes_list_feature(seg_data)),
            'image/segmentation/class/format': _bytes_list_feature('png'),
        }))

        if not os.path.exists('./test_record'):
            os.mkdir('./test_record')
        with tf.io.TFRecordWriter('./test_record/val-test.record') as writer:
            writer.write(example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow', {'VOCRecord':{'root':'./test_record'}}, {'ParseDecodeVoc':{}}, None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (100,100,1))

        from lpot.experimental.data.transforms.transform import ParseDecodeVocTransform
        func = ParseDecodeVocTransform()
        out = func(example.SerializeToString())
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (100,100,3))

        os.remove('./test_record/val-test.record')
        os.remove('test.jpg')
        os.remove('test.png')
        shutil.rmtree('./test_record')

if __name__ == "__main__":
    unittest.main()
