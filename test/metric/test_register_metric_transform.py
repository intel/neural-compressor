"""Tests for neural_compressor register metric and postprocess."""

import os
import platform
import re
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


def build_benchmark():
    seq = [
        "from argparse import ArgumentParser\n",
        "arg_parser = ArgumentParser(description='Parse args')\n",
        "arg_parser.add_argument('--input_model', dest='input_model', default='input_model', help='input model')\n",
        "args = arg_parser.parse_args()\n",
        "import os\n",
        "import numpy as np\n",
        "import PIL.Image\n",
        "image = np.array(PIL.Image.open('images/cat.jpg'))\n",
        "resize_image = np.resize(image, (224, 224, 3))\n",
        "mean = [123.68, 116.78, 103.94]\n",
        "resize_image = resize_image - mean\n",
        "images = np.expand_dims(resize_image, axis=0)\n",
        "labels = [768]\n",
        "from neural_compressor.data.transforms.imagenet_transform import LabelShift\n",
        "from neural_compressor.experimental import Benchmark, common\n",
        "from neural_compressor.experimental.common import Metric, Postprocess\n",
        "from neural_compressor.metric import TensorflowTopK\n",
        "os.environ['NC_ENV_CONF'] = 'True'\n",
        "evaluator = Benchmark('fake_yaml.yaml')\n",
        "nc_postprocess = Postprocess(LabelShift, 'label_benchmark', label_shift=1)\n",
        "evaluator.postprocess = nc_postprocess\n",
        "nc_metric = Metric(TensorflowTopK, 'topk_benchmark')\n",
        "evaluator.metric = nc_metric\n",
        "evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))\n",
        "evaluator.model = args.input_model\n",
        "evaluator.fit()\n",
    ]

    with open("fake.py", "w", encoding="utf-8") as f:
        f.writelines(seq)


def build_benchmark2():
    seq = [
        "from argparse import ArgumentParser\n",
        "arg_parser = ArgumentParser(description='Parse args')\n",
        "arg_parser.add_argument('--input_model', dest='input_model', default='input_model', help='input model')\n",
        "args = arg_parser.parse_args()\n",
        "import os\n",
        "import numpy as np\n",
        "import PIL.Image\n",
        "image = np.array(PIL.Image.open('images/cat.jpg'))\n",
        "resize_image = np.resize(image, (224, 224, 3))\n",
        "mean = [123.68, 116.78, 103.94]\n",
        "resize_image = resize_image - mean\n",
        "images = np.expand_dims(resize_image, axis=0)\n",
        "labels = [768]\n",
        "from neural_compressor.data.transforms.imagenet_transform import LabelShift\n",
        "from neural_compressor.experimental import Benchmark, common\n",
        "from neural_compressor.experimental.common import Metric, Postprocess\n",
        "from neural_compressor.metric import TensorflowTopK\n",
        "os.environ['NC_ENV_CONF'] = 'True'\n",
        "evaluator = Benchmark('fake_yaml.yaml')\n",
        "nc_metric = Metric(TensorflowTopK, 'topk_second')\n",
        "evaluator.metric = nc_metric\n",
        "evaluator.b_dataloader = common.DataLoader(dataset=list(zip(images, labels)))\n",
        "evaluator.model = args.input_model\n\n",
        "evaluator.fit()\n",
    ]

    with open("fake2.py", "w", encoding="utf-8") as f:
        f.writelines(seq)


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
        build_benchmark()
        build_benchmark2()
        if not os.path.exists(self.pb_path) and self.platform == "linux":
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(self.model_url, self.pb_path))

    @classmethod
    def tearDownClass(self):
        if os.path.exists("fake.py"):
            os.remove("fake.py")
        if os.path.exists("fake2.py"):
            os.remove("fake2.py")

    def test_register_metric_postprocess(self):
        os.system("python fake.py --input_model={} 2>&1 | tee benchmark.log".format(self.pb_path))
        with open("benchmark.log", "r") as f:
            for line in f:
                throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)
        os.system("rm benchmark.log")

        os.system("python fake2.py --input_model={} 2>&1 | tee benchmark.log".format(self.pb_path))
        with open("benchmark.log", "r") as f:
            for line in f:
                throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)
        os.system("rm benchmark.log")


if __name__ == "__main__":
    unittest.main()
