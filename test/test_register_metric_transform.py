"""Tests for lpot register metric and postprocess """
import numpy as np
import unittest
import os
import yaml
     
def build_fake_yaml():
    fake_yaml = '''
        model:
          name: resnet_v1_101
          framework: tensorflow
          inputs: input
          outputs: resnet_v1_101/predictions/Reshape_1
        device: cpu
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()


class TestRegisterMetric(unittest.TestCase):
    model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb'
    pb_path = '/tmp/.lpot/resnet101_fp32_pretrained_model.pb'
    #image_path = 'images/1024px-Doll_face_silver_Persian.jpg'
    image_path = 'images/cat.jpg'

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        if not os.path.exists(self.pb_path):
            os.system("mkdir -p /tmp/.lpot && wget {} -O {}".format(self.model_url, self.pb_path))

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    def test_register_metric_postprocess(self):
        import PIL.Image 
        image = np.array(PIL.Image.open(self.image_path))
        resize_image = np.resize(image, (224, 224, 3))
        mean = [123.68, 116.78, 103.94]
        resize_image = resize_image - mean
        images = np.expand_dims(resize_image, axis=0)
        labels = [768]
        from lpot import Benchmark, Quantization, common
        from lpot.data.transforms.imagenet_transform import LabelShift
        from lpot.metric.metric import TensorflowTopK

        evaluator = Benchmark('fake_yaml.yaml')
        evaluator.postprocess = common.Postprocess(LabelShift, 'label_benchmark', label_shift=1) 
        evaluator.metric = common.Metric(TensorflowTopK, 'topk_benchmark')
        evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))
        evaluator.model = self.pb_path
        result = evaluator()
        acc, batch_size, result_list = result['accuracy']
        self.assertEqual(acc, 0.0)

        quantizer = Quantization('fake_yaml.yaml')
        quantizer.postprocess = common.Postprocess(LabelShift, 'label_quantize', label_shift=1) 
        quantizer.metric = common.Metric(TensorflowTopK, 'topk_quantize')

        evaluator = Benchmark('fake_yaml.yaml')
        evaluator.metric = common.Metric(TensorflowTopK, 'topk_second')

        evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))
        evaluator.model = self.pb_path
        result = evaluator()
        acc, batch_size, result_list = result['accuracy']
        self.assertEqual(acc, 0.0)

        

if __name__ == "__main__":
    unittest.main()
