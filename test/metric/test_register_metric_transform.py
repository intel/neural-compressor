"""Tests for neural_compressor register metric and postprocess."""
import os
import platform
import unittest

import numpy as np
import yaml


def build_fake_yaml():
    fake_yaml = """
        model:
          name: resnet_v1_101
          framework: tensorflow
          inputs: input
          outputs: resnet_v1_101/predictions/Reshape_1
        device: cpu
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestRegisterMetric(unittest.TestCase):
    model_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb"
    )
    pb_path = "/tmp/.neural_compressor/resnet101_fp32_pretrained_model.pb"
    # image_path = 'images/1024px-Doll_face_silver_Persian.jpg'
    image_path = "images/cat.jpg"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\.neural_compressor\\resnet101_fp32_pretrained_model.pb"

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        if not os.path.exists(self.pb_path) and self.platform == "linux":
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(self.model_url, self.pb_path))

    def test_register_metric_postprocess(self):
        import PIL.Image

        image = np.array(PIL.Image.open(self.image_path))
        resize_image = np.resize(image, (224, 224, 3))
        mean = [123.68, 116.78, 103.94]
        resize_image = resize_image - mean
        images = np.expand_dims(resize_image, axis=0)
        labels = [768]
        from neural_compressor.data.transforms.imagenet_transform import LabelShift
        from neural_compressor.experimental import Benchmark, common
        from neural_compressor.experimental.common import Metric, Postprocess
        from neural_compressor.metric import TensorflowTopK

        os.environ["NC_ENV_CONF"] = "True"

        evaluator = Benchmark("fake_yaml.yaml")
        nc_postprocess = Postprocess(LabelShift, "label_benchmark", label_shift=1)
        evaluator.postprocess = nc_postprocess
        nc_metric = Metric(TensorflowTopK, "topk_benchmark")
        evaluator.metric = nc_metric
        evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))
        evaluator.model = self.pb_path
        evaluator.fit()

        evaluator = Benchmark("fake_yaml.yaml")
        nc_metric = Metric(TensorflowTopK, "topk_second")
        evaluator.metric = nc_metric
        evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))
        evaluator.model = self.pb_path
        evaluator.fit()


if __name__ == "__main__":
    unittest.main()
