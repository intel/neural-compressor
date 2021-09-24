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
import yaml
import logging

from neural_compressor.adaptor.adaptor import adaptor_registry, Adaptor
from neural_compressor.adaptor.query import QueryBackendCapability
from neural_compressor.utils.utility import dump_elapsed_time, LazyImport, singleton
from collections import OrderedDict
from neural_compressor.adaptor.mxnet_utils.util import *
from copy import deepcopy

mx = LazyImport("mxnet")
logger = logging.getLogger()


@adaptor_registry
class MxNetAdaptor(Adaptor):
    """The MXNet adaptor layer, do MXNet quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super(MxNetAdaptor, self).__init__(framework_specific_info)
        assert check_mx_version('1.6.0'), \
            "Need MXNet version >= 1.6.0, but got version: %s" % (mx.__version__)

        self.pre_optimized_model = None
        self.quantizable_nodes = []
        self._qtensor_to_tensor = {}
        self._tensor_to_node = {}
        self.qdataloader = framework_specific_info.get("q_dataloader")
        self.query_handler = MXNetQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "mxnet.yaml"))

        self.ctx = mx.cpu() if framework_specific_info['device'] == 'cpu' else None
        assert self.ctx is not None, 'Unsupported device'

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, nc_model, dataloader, q_func=None):
        """The function is used to do MXNet calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            nc_model (object): neural_compressor fp32 model to be quantized.
            dataloader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for MXNet.

        Returns:
            (MXNetModel): quantized model
        """
        assert q_func is None, "quantization aware training mode is not supported on mxnet"

        quant_cfg, calib_cfg = parse_tune_config(tune_cfg, self.quantizable_nodes)
        logger.debug("Dump quantization configurations:")
        logger.debug(quant_cfg)

        calib_cache = nc_model.calib_cache

        sym_model, calib_data = prepare_model_data(nc_model, self.ctx, dataloader)
        qsym_model, calib_tensors = quantize_sym_model(sym_model, self.ctx, quant_cfg)
        collector = self._collect_thresholds(sym_model, calib_data, calib_tensors, calib_cfg,
                                             calib_cache)
        qsym_model = calib_model(qsym_model, collector, calib_cfg, logger)
        qsym_model = fuse(qsym_model, self.ctx)  # post-quantization fusion

        q_nc_model = make_nc_model(nc_model, qsym_model, self.ctx, calib_data.input_desc)
        q_nc_model.calib_cache['last'] = collector.th_dict
        q_nc_model.q_config = {
            'mxnet_version': mx.__version__,
            'quant_cfg': quant_cfg,
            'calib_cfg': calib_cfg,
            'th_dict': collector.th_dict,
            'input_desc': calib_data.input_desc,
            'framework_specific_info': {'device': self.ctx.device_type}}

        return q_nc_model

    def _collect_thresholds(self, sym_model, calib_data, calib_tensors, calib_cfg, calib_cache):
        """Calculate thresholds for each tensor. The calibration method can be min/max
           or KL on different tensors.

        Args:
            sym_model (tuple): symbol model (symnet, args, auxs).
            calib_data (DataLoaderWrap): dataset to do calibration on.
            calib_tensors (list): tensors to calibrate
            calib_cfg (dict): calibration config.
            calib_cache (dict): cached calibration results

        Returns:
            collector (CalibCollector): collector with thresholds for each tensor.
        """
        assert calib_cfg['calib_mode'] == 'naive', \
            '`calib_mode` must be set to `naive`, for `collector.min_max_dict` to be used'

        if calib_cache.get('batches', -1) != calib_cfg['batches']:
            calib_cache['batches'] = calib_cfg['batches']
            calib_cache['kl'] = {}
            calib_cache['minmax'] = {}

        cache_kl = calib_cache['kl']
        cache_minmax = calib_cache['minmax']
        tensors_kl, tensors_minmax = distribute_calib_tensors(calib_tensors, calib_cfg,
                                                              self._tensor_to_node)
        to_collect_kl = tensors_kl - set(cache_kl.keys())
        to_collect_minmax = tensors_minmax - set(cache_minmax.keys())
        collector = CalibCollector(to_collect_kl, to_collect_minmax)

        if len(to_collect_kl) + len(to_collect_minmax) > 0:
            def b_filter():
                for _ in range(calib_cfg['batches']):
                    yield True

            logger.info("Start to collect tensors of the FP32 model.")
            batches = run_forward(sym_model, self.ctx, calib_data, b_filter(), collector)
            logger.info("Get collected tensors of the FP32 model from {} batches.".format(batches))

            if len(collector.include_tensors_kl) > 0:
                cache_kl.update(collector.calc_kl_th_dict(calib_cfg['quantized_dtype'],
                                                          logger))
            cache_minmax.update(collector.min_max_dict)

        th_dict = {}
        th_dict.update({tensor: cache_kl[tensor] for tensor in tensors_kl})
        th_dict.update({tensor: cache_minmax[tensor] for tensor in tensors_minmax})
        collector.th_dict = th_dict
        return collector

    def evaluate(self, nc_model, data_x, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """The function is used to run evaluation on validation dataset.

        Args:
            nc_model (object): model to evaluate.
            data_x (object): data iterator/loader.
            postprocess (object, optional): process the result from the model
            metric (metric object): evaluate metric.
            measurer (object, optional): for precise benchmark measurement.
            iteration(int, optional): control steps of mini-batch
            tensorboard (boolean, optional): for tensorboard inspect tensor.
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            acc: evaluate result.
        """
        def b_filter():
            if iteration == -1:
                while True:
                    yield True
            for _ in range(iteration):
                yield True

        def pre_batch(net, batch):
            if measurer is not None:
                measurer.start()

        def post_batch(net, batch, outs):
            if measurer is not None:
                measurer.end()
            _, labels = batch
            outs = ensure_list(outs)
            labels = ensure_list(labels)
            assert len(labels) == len(outs) == 1

            out = outs[0].asnumpy()
            label = labels[0].asnumpy()
            if postprocess is not None:
                out, label = postprocess((out, label))
            if metric is not None:
                metric.update(out, label)

        sym_model, dataloader = prepare_model_data(nc_model, self.ctx, data_x)
        run_forward(sym_model, self.ctx, dataloader, b_filter(),
                    pre_batch=pre_batch, post_batch=post_batch)
        return metric.result() if metric is not None else 0

    @dump_elapsed_time('Query quantizable operators')
    def query_fw_capability(self, nc_model):
        """Query MXNet quantization capability on the model/op level with the specific model.

        Args:
            nc_model (object): model to query.

        Returns:
            dict: modelwise and opwise config.
        """
        # op_type_wise and op_wise capability
        sym_model, self.qdataloader = prepare_model_data(nc_model, self.ctx,
                                                         self.qdataloader)
        self.quantizable_nodes, self._tensor_to_node = query_quantizable_nodes(
            sym_model, self.ctx, self.qdataloader)

        op_type_wise = OrderedDict()
        op_wise = OrderedDict()
        config = self.query_handler.get_quantization_capability()['int8']
        # (TODO) to allign with other fw, set pre_optimized_model here
        self.pre_optimized_model = sym_model
        for node in self.quantizable_nodes:
            optype = node['type']
            op_capability = config.get(optype, config['default'])
            op_type_wise.setdefault(optype, op_capability)
            op_wise.setdefault((node['name'], node['type']), op_capability)

        return {'optypewise': op_type_wise, 'opwise': op_wise}

    def _inspect_tensor(self, nc_model, data_x, node_list=[], iteration_list=[]):
        def b_filter():
            iteration_set = set(iteration_list)
            if len(iteration_set) == 0:
                while True:
                    yield True
            i = 1
            while len(iteration_set) > 0:
                run = (i in iteration_list)
                iteration_set -= {i}
                i += 1
                yield run

        sym_model, dataloader = prepare_model_data(nc_model, self.ctx, data_x)
        collector = TensorCollector(node_list, self._qtensor_to_tensor, self._tensor_to_node)
        num_batches = run_forward(sym_model, self.ctx, dataloader, b_filter(),
                                  collector, pre_batch=collector.pre_batch)
        logger.debug("Inspect batches at {}.".format(num_batches))
        self._qtensor_to_tensor = collector.qtensor_to_tensor
        return collector.tensors_dicts

    def inspect_tensor(self, nc_model, data_x, op_list=[], iteration_list=[],
                       inspect_type='activation', save_to_disk=False):
        """The function is used by tune strategy class for dumping tensor info.

        Args:
            nc_model (object): The model to do calibration.
            data_x (object): Data iterator/loader.
            op_list (list): list of inspect tensors.
            iteration_list (list): list of inspect iterations.

        Returns:
            dict: includes tensor dicts
        """
        if inspect_type not in ['all', 'activation']:
            raise NotImplementedError()

        tensor_dict_list = self._inspect_tensor(nc_model, data_x, op_list, iteration_list)
        for tensor_dict in tensor_dict_list:
            for key, tensors in tensor_dict.items():
                for tensor_name, (is_quantized, tensor) in tensors.items():
                    tensor_dict[key][tensor_name] = tensor  # discard is_quantized
                    if is_quantized:
                        assert tensor.dtype in QUANTIZATION_DTYPES
                        assert 'last' in nc_model.calib_cache
                        min_th, max_th = nc_model.calib_cache['last'][tensor_name]
                        tensor_dict[key][tensor_name] = mx.nd.contrib.dequantize(
                            tensor,
                            min_range=mx.nd.array([min_th]).squeeze(),
                            max_range=mx.nd.array([max_th]).squeeze(),
                            out_type='float32')
                    tensor_dict[key][tensor_name] = tensor_dict[key][tensor_name].asnumpy()

                # transform to format expected by neural_compressor (assume only 1 tensor for now)
                node, op = key
                assert len(tensors) == 1, 'Multiple tensors from a single node are not supported'
                tensor = list(tensor_dict[key].values())[0]
                tensor_dict[key] = {node: tensor}

        return {'activation': tensor_dict_list}

    def recover_tuned_model(self, nc_model, q_config):
        """Execute the recover process on the specified model.

            Args:
                tune_cfg (dict): quantization configuration
                nc_model (object): fp32 model
                q_config (dict): recover configuration

            Returns:
                MXNetModel: the quantized model
        """
        if q_config['mxnet_version'] != mx.__version__:
            logger.warning('Attempting to recover a model generated with a different '
                           'version of MXNet ({})'.format(q_config['mxnet_version']))

        sym_model = prepare_model(nc_model, self.ctx, q_config['input_desc'])
        qsym_model, calib_tensors = quantize_sym_model(sym_model, self.ctx, q_config['quant_cfg'])

        collector = CalibCollector([], [])
        collector.th_dict = q_config['th_dict']
        assert set(calib_tensors).issubset(collector.th_dict.keys())

        qsym_model = calib_model(qsym_model, collector, q_config['calib_cfg'], logger)
        qsym_model = fuse(qsym_model, self.ctx)  # post-quantization fusion

        q_nc_model = make_nc_model(nc_model, qsym_model, self.ctx, q_config['input_desc'])
        q_nc_model.calib_cache['last'] = collector.th_dict
        q_nc_model.q_config = q_config
        return q_nc_model

    def set_tensor(self, model, tensor_dict):
        '''The function is used by tune strategy class for setting tensor back to model.

           Args:
               model (object): The model to set tensor. Usually it is quantized model.
               tensor_dict (dict): The tensor dict to set. Note the numpy array contains float
                                   value, adaptor layer has the responsibility to quantize to
                                   int8 or int32 to set into the quantized model if needed.
                                   The dict format is something like:
                                   {
                                     'weight0_name': numpy.array,
                                     'bias0_name': numpy.array,
                                     ...
                                   }
        '''
        raise NotImplementedError

    def save(self, model, path):
        model.save(path)


@singleton
class MXNetQuery(QueryBackendCapability):
    def __init__(self, local_config_file):
        super().__init__()
        self.version = mx.__version__
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check if the format of {} follows Neural Compressor yaml schema.".
                                 format(self.cfg))

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
        """Get the current backend's version string.
        """
        return deepcopy(self.cur_config['version']['name'])

    def get_precisions(self):
        """Get the supported low precisions, e.g ['int8', 'bf16']
        """
        return deepcopy(self.cur_config['precisions']['names'])

    def get_op_types(self):
        """Get the op types for specific backend per low precision.
            e.g {'1.6.0': {'int8': ['Conv2D', 'fully_connected']}}
        """
        return deepcopy(self.cur_config['ops'])

    def get_fuse_patterns(self):
        """Get the fusion patterns for specified op type for every specific precision

        """
        return deepcopy(self.cur_config['patterns'])

    def get_quantization_capability(self):
        """Get the quantization capability of low precision op types.
            e.g, granularity, scheme and etc.

        """
        return deepcopy(self.cur_config['capabilities'])

    def get_mixed_precision_combination(self, unsupported_precisions=None):
        """Get the valid precision combination base on hardware and user' config.
            e.g['fp32', 'bf16', 'int8']
        """
        self.valid_mixed_precision = []
        if self.cur_config['precisions']['valid_mixed_precisions']:
            for single in self.cur_config['precisions']['valid_mixed_precisions']:
                if not unsupported_precisions in single:
                    self.valid_mixed_precision.append(single)
        return self.valid_mixed_precision if self.valid_mixed_precision \
            else list(self.get_precisions().split(','))
