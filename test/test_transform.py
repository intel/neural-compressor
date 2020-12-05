"""Tests for the transform module."""
import numpy as np
import unittest
import os
from ilit.data import TRANSFORMS, DATASETS, DataLoader
from ilit.utils.create_obj_from_config import get_postprocess, get_preprocess, create_dataset
class TestMetrics(unittest.TestCase):
    def setUp(self):
        pass
    def test_tensorflow_2(self):
        image = np.ones([1, 256, 256, 1])
        resize_kwargs = {"size":[224, 224]}
        transforms = TRANSFORMS(framework="tensorflow", process="preprocess")
        resize = transforms['resize'](**resize_kwargs)
        random_crop_kwargs = {"size": [1, 128, 128, 1]}
        random_crop = transforms['random_crop'](**random_crop_kwargs)
        transform_list = [resize, random_crop]
        compose = transforms['Compose'](transform_list)
        image_result = compose(image)
        self.assertEqual(image_result.shape, (1, 128, 128, 1))


class TestONNXImagenetTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.img = np.random.random_sample([600,600,3])*255
        cls.PIL_img = Image.fromarray(cls.img.astype(np.uint8))

    def testResizeCropImagenetTransform(self):
        transforms = TRANSFORMS('onnx', "preprocess")
        transform = transforms['ResizeCropImagenet'](height=224, width=224)
        sample = (self.PIL_img, 0)
        result = transform(sample)
        resized_input = result[0]
        self.assertEqual(len(resized_input), 3)
        self.assertEqual(len(resized_input[0]), 224)
        self.assertEqual(len(resized_input[0][0]), 224)


class TestTensorflowImagenetTransform(unittest.TestCase):
    def setUp(self):
        pass

    def testBilinearImagenetTransform(self):
        import tensorflow as tf
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
        import tensorflow as tf
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
        import tensorflow as tf
        transforms = TRANSFORMS('tensorflow', "postprocess")
        transform = transforms['LabelShift'](label_shift=1)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 1001)
        label = transform(sample)[1]
        self.assertEqual(label, 1000)

    def testQuantizedInput(self):
        import tensorflow as tf
        transforms = TRANSFORMS('tensorflow', "preprocess")
        transform = transforms['QuantizedInput'](dtype='uint8', scale=100)
        rand_input = np.random.random_sample([600,600,3]).astype(np.float32)
        sample = (rand_input, 1001)
        result = transform(sample)
        quantized_input = result[0].eval(session=tf.compat.v1.Session())
        self.assertLessEqual(quantized_input.max(), 255)
        self.assertGreaterEqual(quantized_input.min(), 0)


class TestTensorflowCOCOTransform(unittest.TestCase):
    def setUp(self):
        pass
    def testPreds(self):
        postprocesses = TRANSFORMS('tensorflow', "postprocess")
        transform = get_postprocess(postprocesses, {'COCOPreds': {}})
        sample = []
        preds = []
        preds.append(np.array([1]))
        preds.append(
            [np.array([[0.16117382, 0.59801614, 0.81511605, 0.7858219 ],
                      [0.5589304 , 0.        , 0.98301625, 0.520178  ]])])
        preds.append([np.array([0.9267181 , 0.8510787])])
        preds.append([np.array([ 1., 67.])])
        sample.append(preds)
        gt = []
        gt.append([np.array([[0.5633255 , 0.34003124, 0.69857144, 0.4009531 ]])])
        gt.append([np.array(['person'.encode('utf-8')])])
        gt.append(['000000397133.jpg'.encode('utf-8')])
        sample.append(gt)
        result = transform(sample)

        self.assertCountEqual(['boxes', 'scores', 'classes'], result[0].keys())
        self.assertCountEqual(['boxes', 'classes'], result[1][0].keys())

        self.assertEqual(len(result[0]['boxes']), len(result[0]['scores']))
        self.assertEqual(len(result[0]['boxes']), len(result[0]['classes']))
        self.assertEqual(len(result[1][0]['boxes']), len(result[1][0]['classes']))
    
    def testCOCODecode(self):
        import tensorflow as tf
        from PIL import Image
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
            'image/object/bbox/xmax':tf.train.Feature(float_list=tf.train.FloatList(value=[100])),
            'image/object/bbox/ymax':tf.train.Feature(float_list=tf.train.FloatList(value=[100])),
        }))

        with tf.io.TFRecordWriter('test.record') as writer:
            writer.write(example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow', {'COCORecord':{'root':'test.record'}}, {'ParseDecodeCoco':{}})
        dataloader = DataLoader(dataset=eval_dataset, framework='tensorflow', batch_size=1)
        for idx, (inputs, labels) in enumerate(dataloader):
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,1,4))
        os.remove('test.record')
        os.remove('test.jpeg')

if __name__ == "__main__":
    unittest.main()
