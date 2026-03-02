#
#  -*- coding: utf-8 -*-
#
import logging
import os
import platform
import sys
import unittest
from importlib.abc import MetaPathFinder

import cpuinfo
import tensorflow as tf


class TestTensorflowGpu(unittest.TestCase):
    mb_model_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb"
    )
    pb_path = "/tmp/.neural_compressor/mobilenet_fp32.pb"
    platforms = platform.system().lower()
    if platforms == "windows":
        pb_path = "C:\\tmp\\.neural_compressor\\mobilenet_fp32.pb"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.pb_path):
            if cls.platforms == "linux":
                os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(cls.mb_model_url, cls.pb_path))
            elif cls.platforms == "windows":
                os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                from urllib import request

                request.urlretrieve(cls.mb_model_url)
        cls.log_env = os.environ.get("LOGLEVEL")
        cls.logger_root = logging.getLogger()
        cls.logger_nc = logging.getLogger("neural_compressor")
        cls.logger_root.setLevel(logging.CRITICAL)
        cls.logger_nc.setLevel(logging.DEBUG)
        cls.logger_root_level = cls.logger_root.level
        cls.logger_nc.warning(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
        cls.logger_nc.warning(f"Environment variable: LOGLEVEL = {cls.log_env}")
        cls.logger_nc.warning(
            f"Before importing neural_compressor: {sys.modules[__name__].__file__}-{cls.__name__}, "
            f"Root_Logger_Level = {cls.logger_root.level}"
        )
        cls.logger_nc.warning(
            f"Before importing neural_compressor: {sys.modules[__name__].__file__}-{cls.__name__}, "
            f"NC_Logger_Level = {cls.logger_nc.level}"
        )
        import neural_compressor
        from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery

        cls.op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

        cls.logger_nc.warning(
            f"After importing neural_compressor: {sys.modules[__name__].__file__}-{cls.__name__}, "
            f"Root_Logger_Level = {cls.logger_root.level}"
        )
        cls.logger_nc.warning(
            f"After importing neural_compressor: {sys.modules[__name__].__file__}-{cls.__name__}, "
            f"NC_Logger_Level = {cls.logger_nc.level}"
        )

    def test_tensorflow_gpu_conversion(self):
        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.int8.post_hostconst_converter import (
            PostHostConstConverter,
        )
        from neural_compressor.tensorflow.quantization.utils.quantize_graph.quantize_graph_for_intel_cpu import (
            QuantizeGraphForIntel,
        )
        from neural_compressor.tensorflow.quantization.utils.utility import read_graph

        input_graph_def = read_graph(self.pb_path)
        input_node_names = ["Placeholder"]
        output_node_names = ["MobilenetV1/Predictions/Reshape_1"]
        op_wise_config = {"MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D": (False, "minmax", False, 7.0)}
        tf.compat.v1.disable_eager_execution()

        converter = QuantizeGraphForIntel(
            input_graph_def, input_node_names, output_node_names, op_wise_config, self.op_wise_sequences, "gpu"
        )
        converted_pb, _, _ = converter.do_transform()
        hostconst_pb = PostHostConstConverter(converted_pb).do_transformation()
        target_node_name = "MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_eightbit_quantized_conv"

        node_details = {}
        for i in hostconst_pb.node:
            node_details[i.name] = i

        converted_flag = True if target_node_name in node_details else False
        self.assertEqual(converted_flag, True)

        target_node = node_details[target_node_name]
        weights_min_node = node_details[target_node.input[-2]]
        weights_max_node = node_details[target_node.input[-1]]
        self.assertEqual(weights_max_node.op, "HostConst")
        self.assertEqual(weights_min_node.op, "HostConst")

        self.assertEqual(self.logger_root.level, self.logger_root_level)


if __name__ == "__main__":
    unittest.main()
