#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import copy
import logging
from collections import OrderedDict
import yaml
import numpy as np
from neural_compressor.adaptor.adaptor import adaptor_registry, Adaptor
from neural_compressor.adaptor.query import QueryBackendCapability
from neural_compressor.utils.utility import LazyImport, dump_elapsed_time
from neural_compressor.utils import logger
from ..utils.utility import OpPrecisionStatistics


@adaptor_registry
class EngineAdaptor(Adaptor):
    """The Engine adaptor layer, do engine quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.work_space = framework_specific_info["workspace_path"]
        os.makedirs(self.work_space, exist_ok=True)
        self.pre_optimized_model = None
        self.query_handler = EngineQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "engine.yaml"))
        self.quantizable_op_types = self._query_quantizable_op_types()
        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.quantize_config = {} # adaptor should know current configs at any time

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            data_loader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for deep engine.

        Returns:
            (dict): quantized model
        """
        assert q_func is None, "quantization aware training has not been supported on Deep engine"
        model = self.pre_optimized_model if self.pre_optimized_model else model
        from neural_compressor.model.engine_model import EngineModel
        tmp_model = EngineModel(model.model)
        self.quantizable_ops = self._query_quantizable_ops(model)
        quantize_config = self._cfg_to_quantize_config(tune_cfg)
        iterations = tune_cfg.get('calib_iteration', 1)
        from neural_compressor.adaptor.engine_utils.engine_quantizer import EngineQuantizer
        quantizer = EngineQuantizer(tmp_model,
            data_loader,
            iterations,
            quantize_config,
            self.quantizable_op_types)
        tmp_model = quantizer.quantize_model()
        self.quantize_config = quantize_config # update so other methods can know current configs
        self._dump_model_op_stastics(tmp_model, quantize_config)
        return tmp_model

    def _dump_model_op_stastics(self, model, config):
        int8_op_list = self.query_handler.get_op_types_by_precision( # pylint: disable=no-member
            precision='int8')
        res = {}
        for op_type in int8_op_list:
            res[op_type] = {'INT8':0, 'BF16': 0, 'FP32':0}
        for node in model.graph.nodes:
            if node.name in config and node.op_type in res:
                if config[node.name] == 'fp32':
                    res[node.op_type]['FP32'] += 1
                else:
                    res[node.op_type]['INT8'] += 1

        output_data = [[op_type, sum(res[op_type].values()), res[op_type]['INT8'],
            res[op_type]['BF16'], res[op_type]['FP32']] for op_type in res.keys()]
        OpPrecisionStatistics(output_data).print_stat()

    def query_fw_capability(self, model):
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: deep engine model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        self.pre_optimized_model = model
        quantizable_ops = self._query_quantizable_ops(self.pre_optimized_model)
        optype_wise = OrderedDict()
        special_config_types = list(self.query_handler.get_quantization_capability()\
                                     ['int8'].keys())  # pylint: disable=no-member
        default_config = self.query_handler.get_quantization_capability()[\
                                     'int8']['default'] # pylint: disable=no-member
        op_wise = OrderedDict()
        for _, op in enumerate(quantizable_ops):
            if op.op_type not in special_config_types:
                op_capability = default_config
            else:
                op_capability = \
                    self.query_handler.get_quantization_capability()[\
                                   'int8'][op.op_type]  # pylint: disable=no-member
            if op.op_type not in optype_wise.keys():
                optype_wise[op.op_type] = copy.deepcopy(op_capability)

            op_wise.update(
                {(op.name, op.op_type): copy.deepcopy(op_capability)})

        return {'optypewise': optype_wise, 'opwise': op_wise}

    def _cfg_to_quantize_config(self, tune_cfg):
        quantize_config = {}
        quantize_config['calib_iteration'] = tune_cfg['calib_iteration']

        for _, op in enumerate(self.quantizable_ops):
            if tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['dtype'] == 'fp32':
                quantize_config[op.name] = 'fp32'
            else:
                node_config = copy.deepcopy(tune_cfg['op'][(op.name, op.op_type)])
                for tensor, config in tune_cfg['op'][(op.name, op.op_type)].items():
                    if config['dtype'] == "int8":
                        node_config[tensor]['dtype'] = 'int8'
                    else:
                        node_config[tensor]['dtype'] = 'uint8'
                quantize_config[op.name] = node_config

        return quantize_config

    def _query_quantizable_ops(self, model):
        for node in model.nodes:
            if node.op_type in self.quantizable_op_types and \
                    node not in self.quantizable_ops:
                self.quantizable_ops.append(node)

        return self.quantizable_ops

    def _query_quantizable_op_types(self):
        quantizable_op_types = self.query_handler.get_op_types_by_precision( \
                                  precision='int8') # pylint: disable=no-member

        return quantizable_op_types

    def evaluate(self, input_graph, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """The function is for evaluation if no given eval func

        Args:
            input_graph      : model for evaluation
            dataloader       : dataloader for evaluation. neural_compressor.data.dataloader.EngineDataLoader
            postprocess      : post-process for evalution. neural_compressor.data.transform.EngineTransforms
            metrics:         : metrics for evaluation. neural_compressor.metric.ONNXMetrics
            measurer         : neural_compressor.objective.Measurer
            iteration(int)   : max iterations of evaluaton.
            tensorboard(bool): whether to use tensorboard for visualizaton
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (float) evaluation results. acc, f1 e.g.
        """
        if metric:
            metric.reset()
            if hasattr(metric, "compare_label") and not metric.compare_label:
                self.fp32_preds_as_label = True
                results = []

        for idx, (inputs, labels) in enumerate(dataloader):
            if measurer is not None:
                measurer.start()
                predictions = input_graph.graph.inference(inputs)
                measurer.end()
            else:
                predictions = input_graph.graph.inference(inputs)
            if self.fp32_preds_as_label:
                self.fp32_results.append(predictions) if fp32_baseline else \
                    results.append(predictions)

            if isinstance(predictions, dict):
                if len(list(predictions.values())) == 1:
                    predictions = list(predictions.values())[0]
                elif len(list(predictions.values())) > 1:
                    predictions = list(predictions.values())[:len(list(predictions.values()))]

            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metric is not None and not self.fp32_preds_as_label:
                metric.update(predictions, labels)
            if idx + 1 == iteration:
                break

        if self.fp32_preds_as_label:
            from neural_compressor.adaptor.engine_utils.util import collate_preds
            if fp32_baseline:
                results = collate_preds(self.fp32_results)
                metric.update(results, results)
            else:
                reference = collate_preds(self.fp32_results)
                results = collate_preds(results)
                metric.update(results, reference)

        acc = metric.result() if metric is not None else 0
        return acc

    def save(self, model, path):
        pass

class EngineQuery(QueryBackendCapability):

    def __init__(self, local_config_file=None):
        super().__init__()
        self.version = 'default'
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e: # pragma: no cover
                logger.info("Failed to parse {} due to {}".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check the {} format.".format(self.cfg))

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        default_config = None
        for sub_data in data:
            if sub_data['version']['name'] == self.version:
                return sub_data

            if sub_data['version']['name'] == 'default':
                default_config = sub_data

        return default_config

    def get_version(self):
        """Get the current backend version infomation.

        Returns:
            [string]: version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self):
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return self.cur_config['precisions']['names']

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config['ops']

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

    def get_op_types_by_precision(self, precision):
        """Get op types per precision

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in list(self.cur_config['ops'].keys())

        return self.cur_config['ops'][precision]
