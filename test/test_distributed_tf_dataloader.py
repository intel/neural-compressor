"""Tests for Distributed TensorFlow Dataloader."""
from neural_compressor import data
from neural_compressor.utils.create_obj_from_config import create_dataset, create_dataloader
from neural_compressor.data.dataloaders.dataloader import DataLoader
from neural_compressor.data import DATASETS, DATALOADERS, TRANSFORMS
import tensorflow as tf
import numpy as np
import collections
import json
import os
import unittest
import shutil

class TestDistributedTFDataDataloader(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.enable_eager_execution()
        self.dataset = tf.data.Dataset.from_tensors((tf.ones([3, 224, 224]), tf.ones([1000]))).repeat(600)
        self.count = 0

    def check_tf_dataset_with_batch_raise(self, batch_size, last_batch, distributed):
        dataset_with_batch = tf.data.Dataset.from_tensors\
            ((tf.ones([3, 224, 224]), tf.ones([1000]))).repeat(600).batch(2)
        dataloader = DATALOADERS['tensorflow']\
            (dataset_with_batch, batch_size=batch_size, last_batch=last_batch, distributed=distributed)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def check_distributed_raise(self, batch_size, last_batch, distributed):
        dataloader = DATALOADERS['tensorflow']\
            (self.dataset, batch_size=batch_size, last_batch=last_batch, distributed=distributed)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_1(self):
        self.assertRaises(TypeError, self.check_tf_dataset_with_batch_raise, 32, 'rollover', True)
    
    def test_dis_tf_data_dataloader_2(self):
        self.assertRaises(TypeError, self.check_tf_dataset_with_batch_raise, 32, 'no_rollover', True)

    def test_dis_tf_data_dataloader_3(self):
        self.assertRaises(TypeError, self.check_tf_dataset_with_batch_raise, 1, 'rollover', True)
    
    def test_dis_tf_data_dataloader_4(self):
        self.assertRaises(TypeError, self.check_tf_dataset_with_batch_raise, 1, 'no_rollover', True)

    def test_dis_tf_data_dataloader_5(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'rollover', True)
    
    def test_dis_tf_data_dataloader_6(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'no_rollover', True)

    def test_dis_tf_data_dataloader_7(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'rollover', True)
    
    def test_dis_tf_data_dataloader_8(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'no_rollover', True)

    def test_dis_tf_data_dataloader_9(self):
        batch_size = 32
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_10(self):
        batch_size = 32
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='no_rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_11(self):
        batch_size = 1
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_12(self):
        batch_size = 1
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='no_rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_13(self):
        batch_size = 600
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_dis_tf_data_dataloader_14(self):
        batch_size = 600
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch='no_rollover')
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (3, 224, 224))
            self.assertEqual(x[-1].shape, (3, 224, 224))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

class TestDefaultDataLoaderSequentialSampler(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if os.path.exists('minist'):
            shutil.rmtree('minist')
    
    def setUp(self):
        self.count = 0

    def check_get_len_raise(self, batch_size, last_batch, distributed):
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': last_batch,
            'distributed': distributed
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        len_dataloader = len(dataloader)

    def check_distributed_raise(self, batch_size, last_batch, distributed):
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': last_batch,
            'distributed': distributed
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler1(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 32, 'rollover', True)

    def test_sequential_sampler2(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 32, 'no_rollover', True)

    def test_sequential_sampler3(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 1, 'rollover', True)

    def test_sequential_sampler4(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 1, 'no_rollover', True)

    def test_sequential_sampler5(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'rollover', True)

    def test_sequential_sampler6(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'no_rollover', True)

    def test_sequential_sampler7(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'rollover', True)

    def test_sequential_sampler8(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'no_rollover', True)

    def test_sequential_sampler9(self):
        batch_size = 3332
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler10(self):
        batch_size = 3332
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'no_rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler11(self):
        batch_size = 1
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler12(self):
        batch_size = 1
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'no_rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler13(self):
        batch_size = 10000
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

    def test_sequential_sampler14(self):
        batch_size = 10000
        dataloader_args = {
            'batch_size': batch_size,
            'dataset': {"MNIST": {'root': './minist', 'train':False, 'download':True}},
            'transform': {'Resize': {'size': 24}},
            'filter': None,
            'last_batch': 'no_rollover'
        }
        dataloader = create_dataloader('tensorflow', dataloader_args)
        for batch in dataloader:
            x, y = batch
            if self.count < len(dataloader)-1:
                self.assertEqual(len(x), batch_size) 
            else:
                self.assertTrue(len(x) == dataloader.dis_dataset_len % batch_size or len(x) == batch_size)
            self.assertEqual(x[0].shape, (24, 24))
            self.assertEqual(x[-1].shape, (24, 24))
            self.assertIsInstance(x, np.ndarray)
            self.count += 1

class TestDefaultDataLoaderIterableSampler(unittest.TestCase):
    class iter_dataset(object):
        def __iter__(self):
            sample_size = 250
            for i in range(1, sample_size+1):
                yield np.array([i])

    def setUp(self):
        self.rank = 0
        self.size = 1 
        self.count = 1
        self.dataset = self.iter_dataset()

    def check_get_len_raise(self, batch_size, last_batch, distributed):
        dataloader = DATALOADERS['tensorflow']\
            (self.dataset, batch_size=batch_size, last_batch=last_batch, distributed=distributed)
        len_dataloader = len(dataloader)

    def check_distributed_raise(self, batch_size, last_batch, distributed):
        dataloader = DATALOADERS['tensorflow']\
            (self.dataset, batch_size=batch_size, last_batch=last_batch, distributed=distributed)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1
    
    def test_iterable_sampler1(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 32, 'rollover', True)

    def test_iterable_sampler2(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 32, 'no_rollover', True)

    def test_iterable_sampler3(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 1, 'rollover', True)

    def test_iterable_sampler4(self):
        self.assertRaises(EnvironmentError, self.check_get_len_raise, 1, 'no_rollover', True)

    def test_iterable_sampler5(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'rollover', True)

    def test_iterable_sampler6(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 32, 'no_rollover', True)

    def test_iterable_sampler7(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'rollover', True)

    def test_iterable_sampler8(self):
        self.assertRaises(EnvironmentError, self.check_distributed_raise, 1, 'no_rollover', True)

    def test_iterable_sampler9(self):
        batch_size = 128
        last_batch = 'rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

    def test_iterable_sampler10(self):
        batch_size = 128
        last_batch = 'no_rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

    def test_iterable_sampler11(self):
        batch_size = 1
        last_batch = 'rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

    def test_iterable_sampler12(self):
        batch_size = 1
        last_batch = 'no_rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

    def test_iterable_sampler13(self):
        batch_size = 1000
        last_batch = 'rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

    def test_iterable_sampler14(self):
        batch_size = 1000
        last_batch = 'no_rollover'
        dataloader = DATALOADERS['tensorflow'](self.dataset, batch_size=batch_size, last_batch=last_batch)
        for batch in dataloader:
            if self.count < len(dataloader):
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(self.count*batch_size*self.size-self.size+self.rank+1, batch[-1][0])
            else:
                self.assertTrue(len(batch) == dataloader.dis_dataset_len % batch_size or len(batch) == batch_size)
                self.assertEqual(((self.count-1)*batch_size+len(batch)-1)*self.size+self.rank+1, batch[-1][0])
                break
            self.count += 1

class TestTensorflowBertDataLoader(unittest.TestCase):
    label = [{
        "paragraphs0":[
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
    unique_id = 1000000000
    input_ids = [101, 2029, 5088, 2136, 3421, 1996, 10511, 2012, 3565, 4605, 2753, 1029, 102, 3565, 4605, 2753,\
        1007, 2005, 1996, 2325, 2161, 1012, 1996, 2137, 2374, 3034, 1006]
    input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    fake_json = json.dumps({'data': label, 'version': '1.1'})
    with open('dev.json', 'w') as f:
        f.write(fake_json)   
    
    @classmethod
    def tearDownClass(cls):
        os.remove('test.record')
        os.remove('dev.json')
    
    def check_not_implement(self, batch_size, distributed):
        with tf.io.TFRecordWriter('./test.record') as writer:
            features = collections.OrderedDict()
            features["unique_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list([self.unique_id])))
            features["input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_ids)))
            features["input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_mask)))
            features["segment_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.segment_ids)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow',
            {'bert':{'root':'test.record', 'label_file': './dev.json'}},
            None,
            None)
        dataloader = DATALOADERS['tensorflow']\
            (dataset=eval_dataset, batch_size=batch_size, distributed=distributed)

    def test_tf_bert_dataloader_1(self):
        self.assertRaises(NotImplementedError, self.check_not_implement, 32, True)
    
    def test_tf_bert_dataloader_2(self):
        batch_size = 128
        with tf.io.TFRecordWriter('./test.record') as writer:
            features = collections.OrderedDict()
            features["unique_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list([self.unique_id])))
            features["input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_ids)))
            features["input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_mask)))
            features["segment_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.segment_ids)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow',
            {'bert':{'root':'test.record', 'label_file': './dev.json'}},
            None,
            None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=batch_size)
        for inputs, labels in dataloader:
            self.assertEqual(inputs[0], 'test.record')
            self.assertEqual(inputs[1], batch_size)
            self.assertEqual(len(labels), 1)

    def test_tf_bert_dataloader_3(self):
        batch_size = 1
        with tf.io.TFRecordWriter('./test.record') as writer:
            features = collections.OrderedDict()
            features["unique_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list([self.unique_id])))
            features["input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_ids)))
            features["input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.input_mask)))
            features["segment_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(self.segment_ids)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        eval_dataset = create_dataset(
            'tensorflow',
            {'bert':{'root':'test.record', 'label_file': './dev.json'}},
            None,
            None)
        dataloader = DATALOADERS['tensorflow'](dataset=eval_dataset, batch_size=batch_size)
        for inputs, labels in dataloader:
            self.assertEqual(inputs[0], 'test.record')
            self.assertEqual(inputs[1], batch_size)
            self.assertEqual(len(labels), 1)

if __name__ == '__main__':
    unittest.main()