"""Tests for the transform module."""
import numpy as np
import unittest
import os
from lpot.data import TRANSFORMS, DataLoader
from lpot.utils.create_obj_from_config import get_postprocess, create_dataset
from lpot.utils.utility import LazyImport
mx = LazyImport('mxnet')
tf = LazyImport('tensorflow')
torchvision = LazyImport('torchvision')

class TestMetrics(unittest.TestCase):
    def test_tensorflow_2(self):
        image = np.ones([1, 256, 256, 1])
        resize_kwargs = {"size":[224, 224]}
        transforms = TRANSFORMS(framework="tensorflow", process="preprocess")
        resize = transforms['Resize'](**resize_kwargs)
        random_crop_kwargs = {"size": [1, 128, 128, 1]}
        random_crop = transforms['RandomCrop'](**random_crop_kwargs)
        transform_list = [resize, random_crop]
        compose = transforms['Compose'](transform_list)
        image_result = compose((image, None))
        self.assertEqual(image_result[0].shape, (1, 128, 128, 1))

class TestONNXQLImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.img = np.random.random_sample([600,600,3])*255
        cls.PIL_img = Image.fromarray(cls.img.astype(np.uint8))

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('onnxrt_qlinearops', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224)
        sample = (self.PIL_img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)

class TestONNXITImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.img = np.random.random_sample([600,600,3])*255
        cls.PIL_img = Image.fromarray(cls.img.astype(np.uint8))

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('onnxrt_integerops', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224)
        sample = (self.PIL_img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)


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
    
    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('tensorflow', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224)
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

class TestSameTransfoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.img = np.random.random_sample([10,10,3])*255
        cls.tf_trans = TRANSFORMS('tensorflow', 'preprocess')
        cls.pt_trans = TRANSFORMS('pytorch', 'preprocess')
        cls.mx_trans = TRANSFORMS('mxnet', 'preprocess')
        cls.ox_trans = TRANSFORMS('onnxrt_qlinearops', 'preprocess')
        cls.mx_img = mx.nd.array(cls.img)
        cls.pt_img = Image.fromarray(cls.img.astype(np.uint8))
        _ = TRANSFORMS('tensorflow', 'postprocess')
        _ = TRANSFORMS('pytorch', 'postprocess')
        _ = TRANSFORMS('mxnet', 'postprocess')
        _ = TRANSFORMS('onnxrt_qlinearops' , 'postprocess')
        _ = TRANSFORMS('onnxrt_integerops', 'postprocess')

    def testCenterCrop(self):
        args = {'size':[4,4]}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['CenterCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CenterCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))
        self.assertEqual(np.array(pt_result)[0][0][0], int(mx_result[0][0][0]))
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))

        args = {'size':4}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['CenterCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['CenterCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))
        self.assertEqual(np.array(pt_result)[0][0][0], int(mx_result[0][0][0]))
        self.assertEqual(np.array(pt_result)[0][0][0], int(tf_result[0][0][0]))
        
        args = {'size':[4]}
        tf_func = TestSameTransfoms.tf_trans['CenterCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4,4,3))

    def testResize(self):
        tf_func = TestSameTransfoms.tf_trans['Resize'](**{'size':[4,5]})
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['Resize'](**{'size':[4,5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Resize'](**{'size':[4,5]})
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,5,3))
        self.assertEqual(pt_result.size, (5,4))
        self.assertEqual(mx_result.shape, (5,4,3))

        pt_func = TestSameTransfoms.pt_trans['Resize'](**{'size':[4,4]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        pt_vision_func = torchvision.transforms.Resize(size=4, interpolation=2)
        pt_vision_result = pt_vision_func(TestSameTransfoms.pt_img)
        self.assertEqual(np.array(pt_result)[0][1][2], np.array(pt_vision_result)[0][1][2])

        args = {'size': 4}
        tf_func = TestSameTransfoms.tf_trans['Resize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['Resize'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['Resize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))

        args = {'size': [4]}
        tf_func = TestSameTransfoms.tf_trans['Resize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        mx_func = TestSameTransfoms.mx_trans['Resize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(mx_result.shape, (4,4,3))

    def testRandomResizedCrop(self):
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**{'size':[4,5]})
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['RandomResizedCrop'](**{'size':[4,5]})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomResizedCrop'](**{'size':[4,5]})
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,5,3))
        self.assertEqual(pt_result.size, (5,4))
        self.assertEqual(mx_result.shape, (5,4,3))

        args = {'size': [4]}
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        self.assertEqual(tf_result.shape, (4,4,3))

        args = {'size': 4}
        tf_func = TestSameTransfoms.tf_trans['RandomResizedCrop'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['RandomResizedCrop'](**args)
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomResizedCrop'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertEqual(tf_result.shape, (4,4,3))
        self.assertEqual(pt_result.size, (4,4))
        self.assertEqual(mx_result.shape, (4,4,3))

    def testCropResize(self):
        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':[5,5]}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))

        args = {'x':0, 'y':0, 'width':10, 'height':10, 'size':5}
        tf_func = TestSameTransfoms.tf_trans['CropResize'](**args)
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        mx_func = TestSameTransfoms.mx_trans['CropResize'](**args)
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        ox_func = TestSameTransfoms.ox_trans['CropResize'](**args)
        ox_result = ox_func((TestSameTransfoms.img, None))[0]
        self.assertEqual(tf_result.shape, (5,5,3))
        self.assertEqual(mx_result.shape, (5,5,3))
        self.assertEqual(ox_result.shape, (5,5,3))

    def testRandomHorizontalFlip(self):
        tf_func = TestSameTransfoms.tf_trans['RandomHorizontalFlip'](**{'seed':1})
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['RandomHorizontalFlip'](**{'p':1.})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomHorizontalFlip']()
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertAlmostEqual(tf_result[0][0][0], mx_result[0][0][0], places=4)
        self.assertEqual(tf_result[0][0][0].astype(np.uint8), np.array(pt_result)[0][0][0])

    def testRandomVerticalFlip(self):
        tf_func = TestSameTransfoms.tf_trans['RandomVerticalFlip'](**{'seed':1})
        tf_result = tf_func((TestSameTransfoms.img, None))
        tf_result = tf_result[0].eval(session=tf.compat.v1.Session())
        pt_func = TestSameTransfoms.pt_trans['RandomVerticalFlip'](**{'p':1.})
        pt_result = pt_func((TestSameTransfoms.pt_img, None))[0]
        mx_func = TestSameTransfoms.mx_trans['RandomVerticalFlip']()
        mx_result = mx_func((TestSameTransfoms.mx_img, None))
        mx_result = mx_result[0].asnumpy()
        self.assertAlmostEqual(tf_result[0][0][0], mx_result[0][0][0], places=4)
        self.assertEqual(tf_result[0][0][0].astype(np.uint8), np.array(pt_result)[0][0][0])

class TestTFTransorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.ones([10,10,3])
        cls.transforms = TRANSFORMS('tensorflow', 'preprocess')

    def testRandomCrop(self):
        args = {'size': [1, 5, 5, 3]}
        transform = TestTFTransorm.transforms['RandomCrop'](**args)
        self.assertRaises(ValueError, transform, (TestTFTransorm.img, None))
        args = {'size': [5, 5, 3]}
        transform = TestTFTransorm.transforms['RandomCrop'](**args)
        img_result = transform((TestTFTransorm.img, None))[0]
        img_result = img_result.eval(session=tf.compat.v1.Session())
        self.assertAlmostEqual(img_result.shape, (5,5,3))

    def testRescale(self):
        transform = TestTFTransorm.transforms['Rescale']()
        img_result = transform((TestTFTransorm.img, None))[0]
        img_result = img_result.eval(session=tf.compat.v1.Session())
        comp_result = np.array(TestTFTransorm.img)/255.
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)

    def testNormalize(self):
        args = {'mean':[0.0,0.0,0.0], 'std':[0.2, 0.5, 0.1]}
        normalize = TestTFTransorm.transforms['Normalize'](**args)
        img_result = normalize((TestTFTransorm.img, None))[0]
        img_result = img_result.eval(session=tf.compat.v1.Session())
        comp_result = np.array(TestTFTransorm.img)/[0.2, 0.5, 0.1]
        self.assertAlmostEqual(img_result[0][0][0], comp_result[0][0][0], places=5)
        self.assertAlmostEqual(img_result[0][0][1], comp_result[0][0][1], places=5)
        self.assertAlmostEqual(img_result[0][0][2], comp_result[0][0][2], places=5)

    def testRandomResizedCrop(self):
        args = {'size':[50]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        image_result = image_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(image_result.shape, (50,50,3))
        args = {'size':[100, 100]}
        randomresizedcrop = TestTFTransorm.transforms["RandomResizedCrop"](**args)
        compose = TestTFTransorm.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestTFTransorm.img, None))[0]
        image_result = image_result.eval(session=tf.compat.v1.Session())
        self.assertEqual(image_result.shape, (100,100,3))
        args = {'size':[100, 100], 'scale':(0.8, 0.1)}
        with self.assertRaises(ValueError):
            TestTFTransorm.transforms["RandomResizedCrop"](**args)


class TestImageTypeParse(unittest.TestCase):
    def testParse(self):
        from PIL import Image
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        img1 = Image.fromarray(random_array)
        onnx_transforms = TRANSFORMS('onnxrt_qlinearops', 'preprocess')
        onnx_parse = onnx_transforms['ImageTypeParse']()
        onnx_compose = onnx_transforms['Compose']([onnx_parse])
        onnx_result = onnx_compose((img1, None))
        self.assertEqual(type(onnx_result[0]).__name__, 'ndarray')

class TestMXnetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random_array = np.random.random_sample([100,100,3]) * 255
        cls.img = mx.nd.array(random_array)
        cls.transforms = TRANSFORMS('mxnet', 'preprocess')

    def testRandomCrop(self):
        args = {'size':[50]}
        randomcrop = TestMXnetTransform.transforms["RandomCrop"](**args)
        compose = TestMXnetTransform.transforms['Compose']([randomcrop])
        image_result = compose((TestMXnetTransform.img, None))
        self.assertEqual(image_result[0].shape, (50,50,3))

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

    def testNormalize(self):
        args = {'mean':[0.0,0.0,0.0], 'std':[0.29, 0.24, 0.25]}
        normalize = TestONNXTransfrom.transforms['Normalize'](**args)
        compose = TestONNXTransfrom.transforms['Compose']([normalize])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertTrue(
            (image_result[0] == np.array(TestONNXTransfrom.img)/[0.29, 0.24, 0.25]).all())

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

    def testCenterCrop(self):
        args = {'size':[50]}
        centercrop = TestONNXTransfrom.transforms["CenterCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([centercrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (50,50,3))

    def testRandomResizedCrop(self):
        args = {'size':[150]}
        randomresizedcrop = TestONNXTransfrom.transforms["RandomResizedCrop"](**args)
        compose = TestONNXTransfrom.transforms['Compose']([randomresizedcrop])
        image_result = compose((TestONNXTransfrom.img, None))
        self.assertEqual(image_result[0].shape, (150,150,3))
        args = {'size':[150, 150], 'scale':(0.9, 0.3), 'interpolation': 'quadratic'}
        with self.assertRaises(ValueError):
            TestONNXTransfrom.transforms["RandomResizedCrop"](**args)

class TestCOCOTransform(unittest.TestCase):
    def testCOCODecode(self):
        from PIL import Image
        from lpot.data.transforms.coco_transform import ParseDecodeCocoTransform
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
            'tensorflow', {'COCORecord':{'root':'test.record'}}, {'ParseDecodeCoco':{}}, None)
        dataloader = DataLoader(dataset=eval_dataset, framework='tensorflow', batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,1,4))

        func = ParseDecodeCocoTransform()
        out = func(example.SerializeToString())
        self.assertEqual(out[0].eval(session=tf.compat.v1.Session()).shape, (100,100,3))
        
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

if __name__ == "__main__":
    unittest.main()
