"""Tests for the dataloader module."""
import unittest
import os
import numpy as np
import shutil
from neural_compressor.utils.create_obj_from_config import create_dataset, create_dataloader
from neural_compressor.data.dataloaders.dataloader import DataLoader
from neural_compressor.data import DATASETS, DATALOADERS, TRANSFORMS
from PIL import Image

class TestBuiltinDataloader(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.remove('./t10k-labels-idx1-ubyte.gz')
        os.remove('./t10k-images-idx3-ubyte.gz')
        os.remove('./train-images-idx3-ubyte.gz')
        os.remove('./train-labels-idx1-ubyte.gz')
        os.remove('./mnist.npz')

    def test_pytorch_dataset(self):
        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR10": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'pytorch', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR100": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'pytorch', dataloader_args)

        dataloader_args = {
            'dataset': {"MNIST": {'root': './test', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'pytorch', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"MNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

        dataloader_args = {
            'batch_size': 2,
            'shuffle': True,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        self.assertEqual(dataloader.dataloader.sampler.__class__.__name__, 'RandomSampler')
        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"FashionMNIST": {'root': './', 'train':True, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'distributed': True
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        self.assertEqual(dataloader.dataloader.sampler.__class__.__name__, 'DistributedSampler')
        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

    def test_mxnet_dataset(self):
        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR10": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'mxnet', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR100": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'mxnet', dataloader_args)

        dataloader_args = {
            'dataset': {"MNIST": {'root': './test', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'mxnet', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"MNIST": {'root': './', 'train':True, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('mxnet', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24,1))
            break

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('mxnet', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24,1))
            break

        dataloader_args = {
            'batch_size': 2,
            'shuffle': True,
            'dataset': {"MNIST": {'root': './', 'train':True, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        with self.assertLogs() as cm:
            dataloader = create_dataloader('mxnet', dataloader_args)
        self.assertEqual(cm.output, ['WARNING:root:Shuffle is not supported yet in' \
                         ' MXNetDataLoader, ignoring shuffle keyword.'])

    def test_tf_dataset(self):
        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR10": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'tensorflow', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR100": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'tensorflow', dataloader_args)

        dataloader_args = {
            'dataset': {"MNIST": {'root': './test', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader, 'tensorflow', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"MNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24))
            break

        dataloader_args = {
            'batch_size': 2,
            'shuffle': True,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        with self.assertLogs() as cm:
            dataloader = create_dataloader('tensorflow', dataloader_args)
        self.assertEqual(cm.output, ['WARNING:root:Shuffle is not supported yet in' \
                         ' TensorflowDataLoader, ignoring shuffle keyword.'])

    def test_onnx_dataset(self):
        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR10": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader,
                            'onnxrt_qlinearops', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"CIFAR100": {'root': './', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader,
                            'onnxrt_qlinearops', dataloader_args)

        dataloader_args = {
            'dataset': {"MNIST": {'root': './test', 'train':False, 'download':False}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        self.assertRaises(RuntimeError, create_dataloader,
                                'onnxrt_qlinearops', dataloader_args)

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"MNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_qlinearops', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24,1))
            break

        dataloader_args = {
            'batch_size': 2,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_qlinearops', dataloader_args)

        for data in dataloader:
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[0][0].shape, (24,24,1))
            break

        dataloader_args = {
            'batch_size': 2,
            'shuffle': True,
            'dataset': {"FashionMNIST": {'root': './', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        with self.assertLogs() as cm:
            dataloader = create_dataloader('onnxrt_qlinearops', dataloader_args)
        self.assertEqual(cm.output, ['WARNING:root:Shuffle is not supported yet' \
                         ' in ONNXRTDataLoader, ignoring shuffle keyword.'])

class TestImagenetRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs('val', exist_ok=True)
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('val/test.jpg')
        with open('val/val.txt', 'w') as f:
            f.write('test.jpg   0')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('val'):
            shutil.rmtree('val')

    def test_tensorflow(self):
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path': './val', 'image_list':None}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/val.txt'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

    def test_pytorch(self):
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path': 'val', 'image_list':None}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/val.txt'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

    def test_mxnet(self):
        import mxnet as mx
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path': 'val', 'image_list':None}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('mxnet', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/val.txt'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('mxnet', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

    def test_onnx(self):
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path': 'val', 'image_list':None}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_integerops', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/val.txt'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_integerops', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break
        # test old api
        eval_dataset = create_dataset('onnxrt_integerops', {'Imagenet':{'root':'./'}}, None, None)
        dataloader = DataLoader('onnxrt_integerops', dataset=eval_dataset, batch_size=1)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (100,100,3))
            break

        with open('val/fake_map.txt', 'w') as f:
            f.write('test.jpg   0 \n')
            f.write('test2.jpg   1')
        dataset_args = {
            "ImagenetRaw": {'data_path':'val', 'image_list':'val/fake_map.txt'},
        }
        dataset = create_dataset('onnxrt_integerops', dataset_args, None, None)
        self.assertEqual(len(dataset), 1)

        with open('val/fake_map.txt', 'w') as f:
            f.write('test2.jpg   1')
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/fake_map.txt'}},
            'transform': None,
            'filter': None
        }
        self.assertRaises(ValueError, create_dataloader, 'onnxrt_integerops', dataloader_args)

        with open('val/not_found_map.txt', 'w') as f:
            f.write('test.jpg   0' + '\n')
            f.write('not_found.jpg   1')
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/not_found_map.txt'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_integerops', dataloader_args)
        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

        with open('val/blank.txt', 'w') as f:
            f.write('blank.jpg   0')
        dataloader_args = {
            'dataset': {"ImagenetRaw": {'data_path':'val', 'image_list':'val/blank.txt'}},
            'transform': None,
            'filter': None
        }
        self.assertRaises(ValueError, create_dataloader,
                            'onnxrt_qlinearops', dataloader_args)


class TestImageFolder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs('val', exist_ok=True)
        os.makedirs('val/0', exist_ok=True)
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('val/0/test.jpg')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('val'):
            shutil.rmtree('val')

    def test_tensorflow(self):
        dataloader_args = {
            'dataset': {"ImageFolder": {'root': './val'}},
            'transform': {'RandomResizedCrop': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)

        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

    def test_pytorch(self):
        dataloader_args = {
            'dataset': {"ImageFolder": {'root': './val'}},
            'transform': {'Resize': {'size': 24}, 'ToTensor':{}},
            'filter': None
        }
        dataloader = create_dataloader('pytorch', dataloader_args)

        for data in dataloader:
            self.assertEqual(data[0][0].shape, (3,24,24))
            break

    def test_mxnet(self):
        dataloader_args = {
            'dataset': {"ImageFolder": {'root': './val'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('mxnet', dataloader_args)

        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

    def test_onnx(self):
        dataloader_args = {
            'dataset': {"ImageFolder": {'root': './val'}},
            'transform': {'Resize': {'size': 24}},
            'filter': None
        }
        dataloader = create_dataloader('onnxrt_integerops', dataloader_args)

        for data in dataloader:
            self.assertEqual(data[0][0].shape, (24,24,3))
            break

class TestDataloader(unittest.TestCase):
    def test_iterable_dataset(self):
        class iter_dataset(object):
            def __iter__(self):
                for i in range(100):
                    yield np.zeros([256, 256, 3])
        dataset = iter_dataset()
        data_loader = DATALOADERS['tensorflow'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))

    def test_tensorflow_bert(self):
        import collections
        import tensorflow as tf
        import json
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
        fake_json = json.dumps({'data': label, 'version': '1.1'})
        with open('dev.json', 'w') as f:
            f.write(fake_json)
        unique_id = 1000000000
        example_index = 0
        doc_span_index = 0
        tokens = ['[SEP]', 'super', 'bowl', '50', 'was', 'an', 'american', 'football', 'game', 'to', 'determine', \
            'the', 'champion', 'of', 'the', 'national', 'football', 'league', '(', 'nfl', ')', 'for', 'the', \
            '2015', 'season', '.', '[SEP]']
        token_to_orig_map = {13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, \
            26: 13, 27: 14, 28: 15, 29: 16, 30: 17, 31: 17, 32: 17, 33: 18, 34: 19, 35: 20, 36: 21, 37: 21, \
            38: 22, 39: 23, 40: 24, 41: 25}
        token_is_max_context = {13: True, 14: True, 15: True, 16: True, 17: True, 18: True, 19: True, 20: True, 21: True, 22: True, \
            23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, 34: True, \
            35: True, 36: True, 37: True, 38: True, 39: True, 40: True, 41: True}
        input_ids = [101, 2029, 5088, 2136, 3421, 1996, 10511, 2012, 3565, 4605, 2753, 1029, 102, 3565, 4605, 2753,\
            1007, 2005, 1996, 2325, 2161, 1012, 1996, 2137, 2374, 3034, 1006]
        input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        with tf.io.TFRecordWriter('./test.record') as writer:
            features = collections.OrderedDict()
            features["unique_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list([unique_id])))
            features["input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(input_ids)))
            features["input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(input_mask)))
            features["segment_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(segment_ids)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow',
            {'bert':{'root':'test.record', 'label_file': './dev.json'}},
            None,
            None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for inputs, labels in dataloader:
            self.assertEqual(inputs[1], 1)
            self.assertEqual(len(labels), 1)
        os.remove('test.record')
        os.remove('dev.json')

    def test_onnx_imagenet(self):
        os.makedirs('val', exist_ok=True)
        os.makedirs('val/0', exist_ok=True)
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('val/test.jpg')
        args = {'ImageFolder': {'root': './val'}}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DATALOADERS['onnxrt_qlinearops'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].size, (100,100))
        shutil.rmtree('val')

    def test_voc_record(self):
        import six
        import collections
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()

        def _bytes_list_feature(values):
            def norm2bytes(value):
                return value.encode() if isinstance(value, str) and six.PY3 else value
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))

        def _int64_list_feature(values):
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

        os.remove('./test_record/val-test.record')
        os.remove('test.jpg')
        os.remove('test.png')
        shutil.rmtree('./test_record')

    def test_coco_record(self):
        import tensorflow as tf
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
            'tensorflow',
            {'COCORecord':{'root':'test.record'}},
            {'RandomVerticalFlip': {},
            'RandomHorizontalFlip': {},
            'CropResize':{'x':0, 'y':0, 'width':10, 'height':10, 'size':[5,5]},
            'Transpose':{'perm': [2, 0, 1]}
            },
            None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for inputs, labels in dataloader:
            self.assertEqual(inputs.shape, (1,3,5,5))
            self.assertEqual(labels[0].shape, (1,1,4))
        os.remove('test.record')
        os.remove('test.jpeg')

    def test_coco_raw(self):
        import json
        import collections
        from neural_compressor.data import TRANSFORMS
        import mxnet as mx
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test_0.jpg')
        im.save('test_1.jpg')
        fake_dict = {
            'info': {
                'description': 'COCO 2017 Dataset',
                'url': 'http://cocodataset.org',
                'version': '1.0',
                'year': 2017,
                'contributor': 'COCO Consortium',
                'date_created': '2017/09/01'
            },
            'licenses':{

            },
            'images':[{
                'file_name': 'test_0.jpg',
                'height': 100,
                'width': 100,
                'id': 0
            },
            {
                'file_name': 'test_1.jpg',
                'height': 100,
                'width': 100,
                'id': 1
            },
            {
                'file_name': 'test_2.jpg',
                'height': 100,
                'width': 100,
                'id': 2
            }],
            'annotations':[{
                'category_id': 18,
                'id': 1767,
                'iscrowd': 0,
                'image_id': 0,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 1,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1769,
                'iscrowd': 0,
                'image_id': 2,
                'bbox': [],
            }],
            'categories':[{
                'supercategory': 'animal',
                'id': 18,
                'name': 'dog'
            }]
        }
        fake_json = json.dumps(fake_dict)
        with open('anno.json', 'w') as f:
            f.write(fake_json)

        args = {'COCORaw': {'root': './', 'img_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('tensorflow', args, None, None)
        dataloader = DATALOADERS['tensorflow'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        trans_args = {'Transpose': {'perm': [2, 0, 1]}}
        ds = create_dataset('tensorflow', args, trans_args, None)
        dataloader = DATALOADERS['tensorflow'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (3,100,100))

        args = {'COCORaw': {'root': './', 'img_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DATALOADERS['onnxrt_qlinearops'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        args = {'COCORaw': {'root': './', 'img_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('mxnet', args, None, None)
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, mx.ndarray.NDArray):
                return mx.nd.stack(*batch)
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, np.ndarray):
                return np.stack(batch)
            else:
                return batch
        dataloader = DATALOADERS['mxnet'](ds, collate_fn=collate)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        args = {'COCORaw': {'root': './', 'img_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('pytorch', args, None, None)
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, np.ndarray):
                return np.stack(batch)
            else:
                return batch
        dataloader = DATALOADERS['pytorch'](dataset=ds, collate_fn=collate)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        os.remove('test_0.jpg')
        os.remove('test_1.jpg')
        os.remove('anno.json')

    def test_coco_npy(self):
        import json
        import collections
        from neural_compressor.data import TRANSFORMS
        import mxnet as mx
        import cv2
        import numpy as np

        def maybe_resize(img, dims):
            img = np.array(img, dtype=np.float32)
            if len(img.shape) < 3 or img.shape[2] != 3:
                # some images might be grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if dims != None:
                im_height, im_width, _ = dims
                img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
            return img

        def convert_npy(src):
            img = cv2.imread(src)
            dims = [100, 100, 3]
            image_format="NHWC"
            need_transpose = True if image_format == "NCHW" else False
            img = maybe_resize(img, dims)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = img / 255. - mean
            img = img / std
            if need_transpose:
                img = img.transpose([2, 0, 1])
            return img

        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test_0.jpg')
        im.save('test_1.jpg')
        im_npy = convert_npy('test_0.jpg')
        np.save('test_0.jpg.npy', im_npy)
        np.save('test_1.jpg.npy', im_npy)

        fake_dict = {
            'info': {
                'description': 'COCO 2017 Dataset',
                'url': 'http://cocodataset.org',
                'version': '1.0',
                'year': 2017,
                'contributor': 'COCO Consortium',
                'date_created': '2017/09/01'
            },
            'licenses':{

            },
            'images':[{
                'file_name': 'test_0.jpg',
                'height': 100,
                'width': 100,
                'id': 0
            },
            {
                'file_name': 'test_1.jpg',
                'height': 100,
                'width': 100,
                'id': 1
            },
            {
                'file_name': 'test_2.jpg',
                'height': 100,
                'width': 100,
                'id': 2
            }],
            'annotations':[{
                'category_id': 18,
                'id': 1767,
                'iscrowd': 0,
                'image_id': 0,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 1,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1769,
                'iscrowd': 0,
                'image_id': 2,
                'bbox': [],
            }],
            'categories':[{
                'supercategory': 'animal',
                'id': 18,
                'name': 'dog'
            }]
        }
        fake_json = json.dumps(fake_dict)
        with open('anno.json', 'w') as f:
            f.write(fake_json)

        args = {'COCONpy': {'root': './', 'npy_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('tensorflow', args, None, None)
        dataloader = DATALOADERS['tensorflow'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        args = {'COCONpy': {'root': './', 'npy_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DATALOADERS['onnxrt_qlinearops'](ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        args = {'COCONpy': {'root': './', 'npy_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('mxnet', args, None, None)
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, mx.ndarray.NDArray):
                return mx.nd.stack(*batch)
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, np.ndarray):
                return np.stack(batch)
            else:
                return batch
        dataloader = DATALOADERS['mxnet'](ds, collate_fn=collate)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        args = {'COCONpy': {'root': './', 'npy_dir': '', 'anno_dir': 'anno.json'}}
        ds = create_dataset('pytorch', args, None, None)
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, np.ndarray):
                return np.stack(batch)
            else:
                return batch
        dataloader = DATALOADERS['pytorch'](dataset=ds, collate_fn=collate)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        os.remove('test_0.jpg')
        os.remove('test_1.jpg')
        os.remove('test_0.jpg.npy')
        os.remove('test_1.jpg.npy')
        os.remove('anno.json')

    def test_tensorflow_imagenet_dataset(self):
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpeg')

        image = tf.compat.v1.gfile.FastGFile('test.jpeg','rb').read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image])),
            'image/class/label':tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[1])),
        }))

        with tf.io.TFRecordWriter('validation-00000-of-00000') as writer:
            writer.write(example.SerializeToString())

        eval_dataset = create_dataset(
            'tensorflow', {'ImageRecord':{'root':'./'}}, None, None)

        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels.shape, (1, 1))

        # test old api
        eval_dataset = create_dataset(
            'tensorflow', {'Imagenet':{'root':'./'}}, None, None)
        dataloader = DataLoader('tensorflow', dataset=eval_dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels.shape, (1, 1))

        os.remove('validation-00000-of-00000')
        os.remove('test.jpeg')

    def test_pytorch_bert_dataset(self):
        dataset = [[
           [101,2043,2001],
           [1,1,1],
           [[0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]],
           [1,1,1],
           [1,1,1],
           [[0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]]
        ]]
        with self.assertRaises(AssertionError):
            create_dataset('pytorch', {'bert': {'dataset':dataset, 'task':'test'}},
                            None, None)

        ds = create_dataset(
            'pytorch',
            {'bert': {'dataset':dataset, 'task':'classifier', 'model_type':'distilbert'}},
            None, None)
        self.assertEqual(len(ds), 1)
        self.assertEqual(3, len(ds[0][0]))

        ds = create_dataset(
            'pytorch',
            {'bert': {'dataset':dataset, 'task':'classifier', 'model_type':'bert'}},
            None, None)
        self.assertEqual(4, len(ds[0][0]))

        ds = create_dataset(
            'pytorch', {'bert': {'dataset':dataset, 'task':'squad'}}, None, None)
        self.assertEqual(3, len(ds[0][0]))

        ds = create_dataset(
            'pytorch',
            {'bert': {'dataset':dataset, 'task':'squad', 'model_type':'distilbert'}},
            None, None)
        self.assertEqual(2, len(ds[0][0]))

        ds = create_dataset(
            'pytorch',
            {'bert': {'dataset':dataset, 'task':'squad', 'model_type':'xlnet'}},
            None, None)
        self.assertEqual(5, len(ds[0][0]))

    def test_tensorflow_dummy(self):
        datasets = DATASETS('tensorflow')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))

        data_loader = DATALOADERS['tensorflow'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

        with self.assertRaises(AssertionError):
            dataset = datasets['dummy'](shape=[(4, 256, 256, 3), (256, 256, 3)])
        with self.assertRaises(AssertionError):
            dataset = datasets['dummy'](shape=(4, 256, 256, 3), low=[1., 0.])
        with self.assertRaises(AssertionError):
            dataset = datasets['dummy'](shape=(4, 256, 256, 3), high=[128., 127.])
        with self.assertRaises(AssertionError):
            dataset = datasets['dummy'](shape=(4, 256, 256, 3), dtype=['float32', 'int8'])

    def test_tensorflow_dummy_v2(self):
        datasets = DATASETS('tensorflow')
        # test with label
        dataset = datasets['dummy_v2'](\
            input_shape=(256, 256, 3), label_shape=(1,))
        data_loader = DATALOADERS['tensorflow'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (1, 256, 256, 3))
        self.assertEqual(data[1].shape, (1, 1))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0].shape, (2, 256, 256, 3))
        self.assertEqual(data[1].shape, (2, 1))

        # test without label
        dataset = datasets['dummy_v2'](input_shape=(256, 256, 3))
        data_loader = DATALOADERS['tensorflow'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

        with self.assertRaises(AssertionError):
            dataset = datasets['dummy_v2'](\
                input_shape=(256, 256, 3), low=[1., 0.])
        with self.assertRaises(AssertionError):
            dataset = datasets['dummy_v2'](\
                input_shape=(256, 256, 3), high=[128., 127.])
        with self.assertRaises(AssertionError):
            dataset = datasets['dummy_v2'](\
                input_shape=(256, 256, 3), dtype=['float32', 'int8'])

    def test_style_transfer_dataset(self):
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpg')

        datasets = DATASETS('tensorflow')
        dataset = datasets['style_transfer'](content_folder='./', style_folder='./')
        length = len(dataset)
        image, label = dataset[0]
        self.assertEqual(image[0].shape, (256, 256, 3))
        self.assertEqual(image[1].shape, (256, 256, 3))
        os.remove('test.jpg')

    def test_tensorflow_list_dict(self):
        dataset = [{'a':1, 'b':2, 'c':3, 'd':4}, {'a':5, 'b':6, 'c':7, 'd':8}]
        data_loader = DATALOADERS['tensorflow'](dataset)

        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1], 'b':[2], 'c':[3], 'd':[4]})

        # test iterable consistent
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1], 'b':[2], 'c':[3], 'd':[4]})

        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1, 5], 'b':[2, 6], 'c':[3, 7], 'd':[4, 8]})

    # def test_tensorflow2_dataset(self):
    #     dataset = [[1, 2, 3, 4], [5, 6, 7, 8]]
    #     dataset = np.array(dataset)
    #     import tensorflow as tf
    #     dataset = tf.data.Dataset.from_tensors(dataset)
    #     data_loader = DATALOADERS['tensorflow'](dataset)

    #     iterator = iter(data_loader)
    #     data = next(iterator)
    #     self.assertEqual(data[0][1], 2)

    def test_pytorch_dummy(self):
        datasets = DATASETS('pytorch')
        transform = TRANSFORMS('pytorch', 'preprocess')['Resize'](**{'size':100})
        dataset = datasets['dummy'](shape=[(4, 256, 256, 3), (4, 1)], \
            high=[10., 10.], low=[0., 0.], transform=transform)

        data_loader = DATALOADERS['pytorch'](dataset)
        iterator = iter(data_loader)
        data, label = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data, label = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_mxnet_dummy(self):
        datasets = DATASETS('mxnet')
        transform = TRANSFORMS('mxnet', 'preprocess')['Resize'](**{'size':100})
        dataset = datasets['dummy'](shape=(4, 256, 256, 3), transform=transform)

        data_loader = DATALOADERS['mxnet'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

        dataset = datasets['dummy'](shape=(4, 256, 256, 3), label=True)
        self.assertEqual(dataset[0][1], 0)

    def test_onnxrt_qlinear_dummy(self):
        datasets = DATASETS('onnxrt_qlinearops')
        transform = TRANSFORMS('onnxrt_qlinearops', 'preprocess')['Resize'](**{'size':100})
        dataset = datasets['dummy'](shape=(4, 256, 256, 3), transform=transform)

        data_loader = DATALOADERS['onnxrt_qlinearops'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_onnx_integer_dummy(self):
        datasets = DATASETS('onnxrt_integerops')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))

        data_loader = DATALOADERS['onnxrt_integerops'](dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_onnx_bert(self):
        import csv
        os.mkdir('./MRPC')
        with open('./MRPC/msr_paraphrase_test.txt', 'a') as f:
            f.write('Quality #1 ID   #2 ID   #1 String   #2 String\n')
            f.write("1   1089874 1089925 PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So . Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So .")
        with open('./MRPC/msr_paraphrase_train.txt', 'a') as f:
            f.write('Quality #1 ID   #2 ID   #1 String   #2 String\n')
            f.write("""1   702876  702977  Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence . Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .""")
        with open('./MRPC/dev.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'])
            tsv_w.writerow(['1', '1355540', '1355592', "He said the foodservice pie business doesn 'tfit thecompany 's long-term growth strategy .", "The foodservice pie businessdoes notfit our long-term growth strategy ."])
        with open('./MRPC/dev_ids.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['1606495', '1606619'])
        with open('./MRPC/test.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['index', '#1 ID', '#2 ID', '#1 String', '#2 String'])
            tsv_w.writerow(['0', '1089874', '1089925', "PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .", "Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So ."])
        with open('./MRPC/train.tsv', 'a') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'])
            tsv_w.writerow(['1', '702876', '702977', """Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .""", """Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence ."""])

        datasets = DATASETS('onnxrt_integerops')
        args = {'GLUE':
                    {'data_dir': './MRPC',
                     'model_name_or_path': 'bert-base-uncased',
                     'dynamic_length': True
                     }}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)

        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DATALOADERS['onnxrt_qlinearops'](ds)
        for inputs, label in dataloader:
            self.assertEqual(len(inputs), 3)
            self.assertEqual(inputs[0].shape[1], 48)
            self.assertEqual(len(label), 1)
            break
        shutil.rmtree('./dataset_cached')

        args = {'GLUE':
                    {'data_dir': './MRPC',
                     'model_type': 'roberta',
                     'model_name_or_path': 'roberta-base',
                     'dynamic_length': False
                     }}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DATALOADERS['onnxrt_qlinearops'](ds)
        for inputs, label in dataloader:
            self.assertEqual(len(inputs), 2)
            self.assertEqual(inputs[0].shape[1], 128)
            self.assertEqual(len(label), 1)
            break

        shutil.rmtree('./MRPC')
        shutil.rmtree('./dataset_cached')


if __name__ == "__main__":
    unittest.main()
