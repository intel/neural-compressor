"""Tests for the dataloader module."""
import unittest
import os
import numpy as np
from lpot.utils.create_obj_from_config import create_dataset
from lpot.data import DATASETS, DataLoader
from PIL import Image

class TestDataloader(unittest.TestCase):
    def test_iterable_dataset(self):
        class iter_dataset(object):
            def __iter__(self):
                for i in range(100):
                    yield np.zeros([256, 256, 3])
        dataset = iter_dataset()
        data_loader = DataLoader('tensorflow', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))

    def test_onnx_imagenet(self):
        import shutil
        os.makedirs('val')
        os.makedirs('val/0')
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('val/0000000397133.jpg')
        args = {'Imagenet': {'root': './'}}
        ds = create_dataset('onnxrt_qlinearops', args, None, None)
        dataloader = DataLoader('onnxrt_qlinearops', ds)
        for image, label in dataloader:
            self.assertEqual(image[0].size, (100,100))
        shutil.rmtree('val')

    def test_coco_raw(self):
        import json
        from lpot.data import TRANSFORMS
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('000000397133.jpg')
        im.save('000000397134.jpg')
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
                'file_name': '000000397133.jpg',
                'height': 100,
                'width': 100,
                'id': 397133
            },
            {
                'file_name': '000000397134.jpg',
                'height': 100,
                'width': 100,
                'id': 397134
            },
            {
                'file_name': '000000397135.jpg',
                'height': 100,
                'width': 100,
                'id': 397135
            }],
            'annotations':[{
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 397133,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 397134,
                'bbox': [473.07, 395.93, 38.65, 28.67],
            },
            {
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 397135,
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
        dataloader = DataLoader('tensorflow', ds)
        for image, label in dataloader:
            self.assertEqual(image[0].size, (100,100))

        trans_args = {'Rescale': {}}
        ds = create_dataset('tensorflow', args, trans_args, None)
        dataloader = DataLoader('tensorflow', ds)
        for image, label in dataloader:
            self.assertEqual(image[0].shape, (100,100,3))

        os.remove('000000397133.jpg')
        os.remove('000000397134.jpg')
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
            'tensorflow', {'Imagenet':{'root':'./'}}, {'ParseDecodeImagenet':{}}, None)
        dataloader = DataLoader(dataset=eval_dataset, framework='tensorflow', batch_size=1) 
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels.shape, (1, 1))

        os.remove('validation-00000-of-00000')
        os.remove('test.jpeg')
        
    def test_tensorflow_dummy(self):
        datasets = DATASETS('tensorflow')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('tensorflow', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_style_transfer_dataset(self):
        jpg_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Doll_face_silver_Persian.jpg/1024px-Doll_face_silver_Persian.jpg"
        os.system("wget {} -O test.jpg".format(jpg_url))
        datasets = DATASETS('tensorflow')
        dataset = datasets['style_transfer'](content_folder='./', style_folder='./')
        length = len(dataset)
        image, label = dataset[0]
        self.assertEqual(image[0].shape, (256, 256, 3))
        self.assertEqual(image[1].shape, (256, 256, 3))
        os.remove('test.jpg')

    def test_tensorflow_list_dict(self):
        dataset = [{'a':1, 'b':2, 'c':3, 'd':4}, {'a':5, 'b':6, 'c':7, 'd':8}]
        data_loader = DataLoader('tensorflow', dataset)

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
    #     data_loader = DataLoader('tensorflow', dataset)
 
    #     iterator = iter(data_loader)
    #     data = next(iterator)
    #     self.assertEqual(data[0][1], 2)
 
    def test_pytorch_dummy(self):
        datasets = DATASETS('pytorch')
        dataset = datasets['dummy'](shape=[(4, 256, 256, 3), (4, 1)], \
            high=[10., 10.], low=[0., 0.])
        
        data_loader = DataLoader('pytorch', dataset)
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
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('mxnet', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_onnxrt_qlinear_dummy(self):
        datasets = DATASETS('onnxrt_qlinearops')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('onnxrt_qlinearops', dataset)
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

        data_loader = DataLoader('onnxrt_integerops', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))


if __name__ == "__main__":
    unittest.main()
