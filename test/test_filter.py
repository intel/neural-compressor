import unittest
import tensorflow as tf
import numpy as np
import os
import json
import shutil
from PIL import Image
from lpot.data import FILTERS, TRANSFORMS, DATASETS, DATALOADERS
from lpot.utils.create_obj_from_config import create_dataset, get_preprocess, create_dataloader

class TestCOCOFilter(unittest.TestCase):
    def testLabelBalanceCOCORecord(self):
        tf.compat.v1.disable_eager_execution() 

        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        im.save('test.jpeg')

        image = tf.compat.v1.gfile.FastGFile('test.jpeg','rb').read()
        source_id = '000000397133.jpg'.encode('utf-8')
        label = 'person'.encode('utf-8')
        example1 = tf.train.Example(features=tf.train.Features(feature={
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
        example2 = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image])),
            'image/object/class/text':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[label])),
            'image/source_id':tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[source_id])),
            'image/object/bbox/xmin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10, 20])),
            'image/object/bbox/ymin':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[10, 20])),
            'image/object/bbox/xmax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100, 200])),
            'image/object/bbox/ymax':tf.train.Feature(
                    float_list=tf.train.FloatList(value=[100, 200])),
        }))
        with tf.io.TFRecordWriter('test.record') as writer:
            writer.write(example1.SerializeToString())
            writer.write(example2.SerializeToString())
        
        preprocesses = TRANSFORMS('tensorflow', 'preprocess')
        filters = FILTERS('tensorflow')
        filter = filters['LabelBalanceCOCORecord'](2)
        datasets = DATASETS('tensorflow')
        dataset = datasets['COCORecord']('test.record', \
            transform=None, filter=filter)
        dataloader = DATALOADERS['tensorflow'](dataset=dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))

        dataset2 = create_dataset(
            'tensorflow', {'COCORecord':{'root':'test.record'}}, None, {'LabelBalance':{'size':2}})
        dataloader2 = DATALOADERS['tensorflow'](dataset=dataset2, batch_size=1)
        for (inputs, labels) in dataloader2:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))

        dataloader3 = create_dataloader('tensorflow', {'batch_size':1, 'dataset':{'COCORecord':{'root':'test.record'}},\
                 'filter':{'LabelBalance':{'size':2}}, 'transform':None})
        for (inputs, labels) in dataloader3:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))
        os.remove('test.record')
        os.remove('test.jpeg')

    def testLabelBalanceCOCORaw(self):
        random_array = np.random.random_sample([100,100,3]) * 255
        random_array = random_array.astype(np.uint8)
        im = Image.fromarray(random_array)
        os.makedirs('val2017', exist_ok=True)
        im.save('./val2017/test_0.jpg')
        im.save('./val2017/test_1.jpg')
        fake_dict = {
            'info': {
                'description': 'COCO 2017 Dataset',
                'url': 'http://cocodataset.org',
                'version': '1.0',
                'year': 2017,
                'contributor': 'COCO Consortium',
                'date_created': '2017/09/01'
            },
            'licenses':{},
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
               'id': 1768,
               'iscrowd': 0,
               'image_id': 1,
               'bbox': [473.07, 395.93, 38.65, 28.67],
            }],
            'categories':[{
                'supercategory': 'animal',
                'id': 18,
                'name': 'dog'
            }]
        }
        fake_json = json.dumps(fake_dict)
        os.makedirs('annotations', exist_ok=True)
        with open('./annotations/instances_val2017.json', 'w') as f:
            f.write(fake_json)

        filters = FILTERS('onnxrt_qlinearops')
        filter = filters['LabelBalanceCOCORaw'](1)
        datasets = DATASETS('onnxrt_qlinearops')
        dataset = datasets['COCORaw']('./', transform=None, filter=filter)
        dataloader = DATALOADERS['onnxrt_qlinearops'](dataset=dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(labels[0].shape[1], 1)

        shutil.rmtree('annotations')
        shutil.rmtree('val2017')

if __name__ == "__main__":
    unittest.main()

