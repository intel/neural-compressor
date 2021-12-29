#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

from __future__ import division
import time
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

class eval_object_detection_optimized_graph(object):
    
    def __init__(self):
        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-g',
                            "--input-graph",
                            help='Specify the input graph.',
                            dest='input_graph')
        arg_parser.add_argument('--config', type=str, default='')
        arg_parser.add_argument('--output_model', type=str, default='')
        arg_parser.add_argument('--mode', type=str, default='performance')
        arg_parser.add_argument('--tune', action='store_true', default=False)
        arg_parser.add_argument('--benchmark', dest='benchmark',
                            action='store_true', help='run benchmark')
        self.args = arg_parser.parse_args()
        
    def run(self):
        if self.args.tune:
            from neural_compressor.experimental import Quantization
            quantizer = Quantization(self.args.config)
            quantizer.model = self.args.input_graph
            q_model = quantizer.fit()
            q_model.save(self.args.output_model)
                
        if self.args.benchmark:
            from neural_compressor.experimental import Benchmark
            evaluator = Benchmark(self.args.config)
            evaluator.model = self.args.input_graph
            evaluator(self.args.mode)

if __name__ == "__main__":
    evaluate_opt_graph = eval_object_detection_optimized_graph()
    evaluate_opt_graph.run()
