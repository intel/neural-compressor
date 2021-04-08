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

    def test_register_metric_postprocess(self):
        import PIL.Image 
        image = np.array(PIL.Image.open(self.image_path))
        resize_image = np.resize(image, (224, 224, 3))
        mean = [123.68, 116.78, 103.94]
        resize_image = resize_image - mean
        images = np.expand_dims(resize_image, axis=0)
        labels = [768]
        from lpot import Benchmark, Quantization
        from lpot.experimental.data.transforms.imagenet_transform import LabelShift
        from lpot.experimental.metric.metric import TensorflowTopK

        evaluator = Benchmark('fake_yaml.yaml')

        evaluator.postprocess('label_benchmark', LabelShift, label_shift=1) 
        evaluator.metric('topk_benchmark', TensorflowTopK)
        # as we supported multi instance, the result will print out instead of return
        dataloader = evaluator.dataloader(dataset=list(zip(images, labels)))
        evaluator(self.pb_path, b_dataloader=dataloader)

        quantizer = Quantization('fake_yaml.yaml')
        quantizer.postprocess('label_quantize', LabelShift, label_shift=1) 
        quantizer.metric('topk_quantize', TensorflowTopK)

        evaluator = Benchmark('fake_yaml.yaml')
        evaluator.metric('topk_second', TensorflowTopK)

        dataloader = evaluator.dataloader(dataset=list(zip(images, labels)))
        result = evaluator(self.pb_path, b_dataloader=dataloader)


if __name__ == "__main__":
    unittest.main()
