import unittest
import tensorflow as tf
import numpy as np
import os
from lpot.data import FILTERS, TRANSFORMS, DATASETS, DATALOADERS
from lpot.utils.create_obj_from_config import create_dataset, get_preprocess, create_dataloader

class TestCOCOFilter(unittest.TestCase):
    def testLabelBalanceCOCORecord(self):
        from PIL import Image
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
        preprocess = get_preprocess(preprocesses, {'ParseDecodeCoco':{}})
        filters = FILTERS('tensorflow')
        filter = filters['LabelBalanceCOCORecord'](2)
        datasets = DATASETS('tensorflow')
        dataset = datasets['COCORecord']('test.record', \
            transform=preprocess, filter=filter)
        dataloader = DATALOADERS['tensorflow'](dataset=dataset, batch_size=1)
        for (inputs, labels) in dataloader:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))

        dataset2 = create_dataset(
            'tensorflow', {'COCORecord':{'root':'test.record'}}, {'ParseDecodeCoco':{}}, {'LabelBalance':{'size':2}})
        dataloader2 = DATALOADERS['tensorflow'](dataset=dataset2, batch_size=1)
        for (inputs, labels) in dataloader2:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))

        dataloader3 = create_dataloader('tensorflow', {'batch_size':1, 'dataset':{'COCORecord':{'root':'test.record'}},\
                 'filter':{'LabelBalance':{'size':2}}, 'transform':{'ParseDecodeCoco':{}}})
        for (inputs, labels) in dataloader3:
            self.assertEqual(inputs.shape, (1,100,100,3))
            self.assertEqual(labels[0].shape, (1,2,4))
        os.remove('test.record')
        os.remove('test.jpeg')


if __name__ == "__main__":
    unittest.main()

