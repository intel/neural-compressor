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
import logging
import json
import re
from tempfile import TemporaryDirectory
import numpy as np
import yaml

from .adaptor import adaptor_registry, Adaptor
from .query import QueryBackendCapability
from ..utils.utility import LazyImport, singleton
from ..utils.kl_divergence import KL_Divergence
from ..utils.collect_layer_histogram import LayerHistogramCollector
from collections import OrderedDict
from lpot.utils.utility import dump_elapsed_time
from lpot.model.model import MXNetModel as Model

mx = LazyImport("mxnet")

logger = logging.getLogger()


def _check_version(v1, v2):
    """version checkout functioin.

    Args:
        v1 (str): first version number.
        v2 (str): sencond version number.

    Returns:
        boolen: True if v1 > v2, else False.
    """
    d1 = re.split(r'\.', v1)
    d2 = re.split(r'\.', v2)
    d1 = [int(d1[i]) for i in range(len(d1))]
    d2 = [int(d2[i]) for i in range(len(d2))]

    if d1 >= d2:
        return True
    return False


@adaptor_registry
class MxNetAdaptor(Adaptor):
    """The MXNet adaptor layer, do MXNet quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super(MxNetAdaptor, self).__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.logger = logger
        self.qdataloader = framework_specific_info["q_dataloader"]
        self.query_handler = MXNetQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "mxnet.yaml"))

        # MXNet version check
        if not _check_version(mx.__version__, '1.6.0'):
            raise Exception("Need MXNet version >= 1.6.0, but get version: %s" % (mx.__version__))

    def _get_backedn_graph(self, symbol, ctx):
        if ctx == mx.cpu():
            return symbol.get_backend_symbol('MKLDNN_QUANTIZE')
        else:
            pass

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """The function is used to do MXNet calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for MXNet.

        Returns:
            (dict): quantized model
        """
        assert q_func is None, "quantization aware training mode is not support on mxnet"

        # get symbol from FP32 model
        if isinstance(model.model, mx.gluon.HybridBlock):
            # transfer hybridblock to symbol
            sym, arg_params, aux_params, calib_data = \
                self._get_gluon_symbol(model.model, dataloader=dataloader)
            data_names = [pair[0] for pair in calib_data.provide_data]
            self.__config_dict['calib_data'] = calib_data
        elif isinstance(model.model[0], mx.symbol.Symbol):
            sym, arg_params, aux_params = model.model
            self.__config_dict['calib_data'] = dataloader
        else:
            raise ValueError(
                'Need a symbol model or HybridBlock model, while received %s' % str(
                    type(model.model)))

        self._cfg_to_qconfig(tune_cfg)
        self.th_dict = None
        qconfig = self.__config_dict
        sym = self._get_backedn_graph(sym, qconfig['ctx'])

        # 1. quantize_symbol
        if _check_version(mx.__version__, '1.7.0'):
            qsym, calib_layer = mx.contrib.quantization._quantize_symbol(
                sym, qconfig['ctx'],
                excluded_symbols=qconfig['excluded_sym_names'],
                excluded_operators=qconfig['excluded_op_names'],
                offline_params=list(arg_params.keys()),
                quantized_dtype=qconfig['quantized_dtype'],
                quantize_mode=qconfig['quantize_mode'],
                quantize_granularity=qconfig['quantize_granularity'])
        else:
            qsym, calib_layer = mx.contrib.quantization._quantize_symbol(
                sym, qconfig['ctx'],
                excluded_symbols=qconfig['excluded_sym_names'],
                excluded_operators=qconfig['excluded_op_names'],
                offline_params=list(arg_params.keys()),
                quantized_dtype=qconfig['quantized_dtype'],
                quantize_mode=qconfig['quantize_mode'],)

        # 2. Do calibration to get th_dict
        th_dict = self._get_calibration_th(
            sym, arg_params, aux_params, calib_layer, qconfig)
        self.th_dict = th_dict
        # 3. set th_dict to quantized symbol
        qsym = mx.contrib.quantization._calibrate_quantized_sym(qsym, th_dict)
        # 4. quantize params
        qarg_params = mx.contrib.quantization._quantize_params(
            qsym, arg_params, th_dict)
        qsym = self._get_backedn_graph(qsym, qconfig['ctx'])

        if isinstance(model.model, mx.gluon.HybridBlock):
            from mxnet.gluon import SymbolBlock
            data_sym = []
            for name in data_names:
                data_sym.append(mx.sym.var(name))
            net = SymbolBlock(qsym, data_sym)

            with TemporaryDirectory() as tmpdirname:
                prefix = os.path.join(tmpdirname, 'tmp')
                param_name = '%s-%04d.params' % (prefix + 'net-quantized', 0)
                save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu())
                             for k, v in qarg_params.items()}
                save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu())
                                  for k, v in aux_params.items()})
                mx.ndarray.save(param_name, save_dict)  # pylint: disable=no-member
                net.collect_params().load(param_name, cast_dtype=True, dtype_source='saved')
                net.collect_params().reset_ctx(self.__config_dict['ctx'])
                return Model(net)
        return Model((qsym, qarg_params, aux_params))

    def train(self, model, dataloader):
        """The function is used to do training in quantization-aware training.

        Args:
            model (object): The model to do calibration.
            dataloader (object): The dataset do do QAT.
        """
        raise NotImplementedError

    def evaluate(self, model, dataloader, postprocess=None, \
                 metric=None, measurer=None, iteration=-1, \
                 tensorboard=False, fp32_baseline=False):
        """The function is used to run evaluation on validation dataset.

        Args:
            model (object): model to do evaluate.
            dataloader (object): dataset to do evaluate.
            postprocess (object, optional): process the result from the model
            metric (metric object): evaluate metric.
            measurer (object, optional): for precise benchmark measurement.
            iteration(int, optional): control steps of mini-batch
            tensorboard (boolean, optional): for tensorboard inspect tensor.
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            acc: evaluate result.
        """
        if isinstance(model.model, mx.gluon.HybridBlock):
            acc = self._mxnet_gluon_forward(model.model, dataloader, postprocess, \
                                            metric, measurer, iteration)

        elif isinstance(model.model[0], mx.symbol.symbol.Symbol):
            assert isinstance(dataloader, mx.io.DataIter), \
                'need mx.io.DataIter. but recived %s' % str(type(dataloader))
            dataloader.reset()
            acc = self._mxnet_symbol_forward(model.model, dataloader, postprocess, \
                                             metric, measurer, iteration)

        else:
            raise ValueError("Unknow graph tyep: %s" % (str(type(model))))

        return acc

    def _mxnet_symbol_forward(self, symbol_file, dataIter, \
                              postprocess, metric, measurer, iteration):
        """MXNet symbol model evaluation process.
        Args:
            symbol_file (object): the symbole model need to do evaluate.
            dataIter (object): dataset used for model evaluate.
            metric (object): evaluate metrics.

        Returns:
            acc: evaluate result
        """
        sym, arg_params, aux_params = symbol_file
        sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
        data_name = (dataIter.provide_data[0].name,)
        label_name = (dataIter.provide_label[0].name,)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(),
                            data_names=data_name,
                            label_names=label_name
                            )
        mod.bind(for_training=False,
                 data_shapes=dataIter.provide_data,
                 label_shapes=dataIter.provide_label
                 )
        mod.set_params(arg_params, aux_params)

        batch_num = 0
        for idx, batch in enumerate(dataIter):
            if measurer is not None:
                measurer.start()
                mod.forward(batch, is_train=False)
                measurer.end()
            else:
                mod.forward(batch, is_train=False)

            output = mod.get_outputs()
            output = output[0].asnumpy()
            label = batch.label[0].asnumpy()
            if postprocess is not None:
                output, label = postprocess((output, label))
            if metric is not None:
                metric.update(output, label)
            batch_num += dataIter.batch_size
            if idx + 1 == iteration:
                break
        acc = metric.result() if metric is not None else 0
        return acc

    def _mxnet_gluon_forward(self, gluon_model, dataloader,
                             postprocess, metric, measurer, iteration):
        """MXNet gluon model evaluation process.

        Args:
            gluon_model (object): the gluon model need to do evaluate.
            dataloader (object): dataset used for model evaluate.
            metrics (object): evaluate metrics.

        Returns:
            acc: evaluate result
        """
        raise NotImplementedError

    def _check_model(self, model, dataloader):
        """The function is used to check model and calib_data,
           if not symbol and dataiter, then transfer it to.

        Args:
            model (object): The model to do calibration.
            dataloader (object): Used to when check a gluon model.
        Returns:
            tuple: Symbol model include sym, arg_params, aux_params.
        """
        from lpot.model.model import MXNetModel
        if isinstance(model, MXNetModel):
            model = model.model
        if isinstance(model, mx.gluon.HybridBlock):
            # model.hybridblock()
            sym, arg_params, aux_params, calib_data = \
                self._get_gluon_symbol(network=model, dataloader=dataloader)
            self.__config_dict['calib_data'] = calib_data
        elif isinstance(model[0], mx.symbol.Symbol):
            sym, arg_params, aux_params = model
            calib_data = dataloader
        else:
            raise TypeError

        return sym, arg_params, aux_params, calib_data

    @dump_elapsed_time("Query quantizable operators")
    def _query_quantizable_ops(self, model, calib_data):
        """Query quantizable ops of the given model.

        Args:
            model (Symbol or HybridBlock): model to query.
            calib_data (DataIter or Dataloader): dataset to do calibration.

        Returns:
            list: quantizable ops of the given model.
        """
        if len(self.quantizable_ops) != 0:
            return self.quantizable_ops

        sym, arg_params, aux_params, calib_data = self._check_model(
            model, calib_data)
        sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

        _, calib_layer = mx.contrib.quantization._quantize_symbol(
            sym,
            mx.cpu(),
            offline_params=list(arg_params.keys()),
        )

        # get each op type
        dct = json.loads(sym.tojson())
        op_type = []
        for item in dct['nodes']:
            op_type.append(item['op'])

        symbol_layers = []
        sym_all_layers = [layer.name for layer in list(sym.get_internals())]
        name_type = zip(sym_all_layers, op_type)

        arg_params_list = list(arg_params.keys())
        aux_params_list = list(aux_params.keys())
        for name, type in name_type:
            if name not in arg_params_list and item not in aux_params_list:
                symbol_layers.append({"name": name, "type": type})

        for _, opname_type in enumerate(symbol_layers):
            if opname_type["name"] + "_output" in calib_layer:
                self.quantizable_ops.append(opname_type)

        return self.quantizable_ops

    def query_fused_patterns(self, model):
        """The function is used to run fused patterns in framework.

        Args:
            model (object): The model to do calibration.
        """
        raise NotImplementedError

    def query_fw_capability(self, model):
        """Query MXNet quantization capability on the model/op level with the specific model.

        Args:
            model (Symbol or HybridBlock): model to query.

        Returns:
            (dict): modelwise and opwise config.
        """
        # op_type_wise and op_wise capability
        quantizable_ops = self._query_quantizable_ops(model, self.qdataloader)
        op_type_wise = OrderedDict()
        op_wise = OrderedDict()
        quantizable_op_config = self.query_handler.get_quantization_capability()['int8']
        mixed_quantization = self.query_handler.get_mixed_precision_combination()
        for _, opname_type in enumerate(quantizable_ops):
            optype = opname_type["type"]
            if optype in quantizable_op_config.keys():
                op_capability = quantizable_op_config[optype]
                if optype not in op_type_wise.keys():
                    op_type_wise[optype] = quantizable_op_config[optype]
            else:
                op_capability = quantizable_op_config['default']
                if optype not in op_type_wise.keys():
                    op_type_wise[optype] = quantizable_op_config['default']

            op_wise.update(
                {(opname_type["name"], opname_type["type"]): op_capability})

        return {'optypewise': op_type_wise, 'opwise': op_wise}

    @dump_elapsed_time("Collect calibration statistics")
    def _inspect_tensor(
            self,
            model,
            dataloader,
            op_list=[],
            iteration_list=[]):
        """The function is used by tune strategy class for dumping tensor info.

        Args:
            model (object): The model to do calibration.
            dataloader (object): The data to do forward.
            op_list (list): list of inspect tensors.
            iteration_list (list): list of inspect iterations.

        Returns:
            Numpy Array Dict
            if iteration_list is empty:
                {'op1':  tensor, ...}
            if iteration_list is not empty:
                {"iteration_1": {'op1': tensor, ...}
                    "iteration_2": {'op1': tensor, ...}}
        """
        import ctypes
        from mxnet.base import _LIB, py_str
        from mxnet.base import NDArrayHandle

        class _LayerTensorCollector(object):
            """Saves layer output min and max values in a dict with layer names as keys.
            The collected min and max values will be directly used as thresholds for quantization.
            """

            def __init__(self, include_layer=None, logger=None):
                self.tensor_dict = {}
                self.include_layer = include_layer
                self.logger = logger

            def collect(self, name, arr):
                """Callback function for collecting min and max values from an NDArray."""
                name = py_str(name)
                if name not in self.include_layer:
                    return

                handle = ctypes.cast(arr, NDArrayHandle)
                arr = mx.ndarray.NDArray(handle, writable=False).asnumpy()  # pylint: disable=no-member
                if name in self.tensor_dict.keys():
                    self.tensor_dict[name].append(arr)
                else:
                    self.tensor_dict[name] = [arr]

            def reset(self,):
                self.tensor_dict.clear()

        # setup collector, only collect quantized layer output
        include_layer = op_list
        collector = _LayerTensorCollector(include_layer=include_layer)
        data = self.__config_dict['calib_data']

        # setup mod
        data_names = [pair[0] for pair in data.provide_data]
        calib_iter = self.__config_dict['iteration']
        sym, arg_params, aux_params = model
        mod = mx.module.module.Module(
            symbol=sym,
            data_names=data_names,
            context=self.__config_dict['ctx'])
        mod.bind(for_training=False, data_shapes=data.provide_data)
        mod.set_params(arg_params, aux_params)
        mod._exec_group.execs[0].set_monitor_callback(
            collector.collect, monitor_all=True)
        num_batches = 0

        if len(iteration_list) == 0:
            for _, batch in enumerate(data):
                mod.forward(data_batch=batch, is_train=False)
                num_batches += 1
                if calib_iter is not None and num_batches >= calib_iter:
                    break

            self.logger.info(
                "Inspect tensors from %d batches with batch_size=%d" %
                (num_batches, data.batch_size))
            return collector.tensor_dict

        else:
            iter_tensor = {}
            num_batches = 0
            for _, batch in enumerate(data):
                if num_batches in iteration_list:
                    iter_name = "iteration_" + str(num_batches)
                    mod.forward(data_batch=batch, is_train=False)
                    iter_value = collector.tensor_dict.copy()
                    if len(iteration_list) == 1:
                        return iter_value
                    iter_tensor.update({iter_name: iter_value})
                    collector.reset()
                num_batches += 1

            self.logger.info(
                "Inspect tensors from %d batches with batch_size=%d" %
                (num_batches, data.batch_size))
            return iter_tensor

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        int8_ops_th = self.th_dict
        op_list_convert = []
        sym, arg_params, aux_params, dataloader = self._check_model(
            model, dataloader)
        sym_all_layers = [layer.name for layer in list(sym.get_internals())]
        for item in op_list:
            op_name = item[0]
            if "quantized_" + op_name in sym_all_layers:
                op_name = "quantized_" + op_name
            if not op_name.endswith("_output"):
                op_name += "_output"
            op_list_convert.append(op_name)
        dataloader.reset()
        inspected_tensor = self._inspect_tensor(
            (sym, arg_params, aux_params), dataloader, op_list_convert, iteration_list)
        inspected_tensor_convert = {}
        for op, tensor in inspected_tensor.items():
            if op.startswith("quantized_"):
                op = op[10:]
                if op in int8_ops_th:
                    op_min = mx.nd.array(int8_ops_th[op][0])
                    op_max = mx.nd.array(int8_ops_th[op][1])
                    # TODO: deal hard code dtype
                    tensor = mx.nd.contrib.dequantize(mx.nd.array(tensor, dtype='uint8'),
                                                      min_range=op_min,
                                                      max_range=op_max,
                                                      out_type='float32')
                    assert tensor.dtype == np.float32
            if op.endswith("_output"):
                op = op[:-7]
            for item in op_list:
                if op in item:
                    op = item
                    break
            inspected_tensor_convert.update({op: tensor})

        return inspected_tensor_convert

    def mapping(self, src_model, dst_model):
        """The function is used to create a dict to map tensor name of src model to tensor name of
           dst model.

        Returns:
            Dict
            {'src_op1': 'dst_op1'}
        """
        raise NotImplementedError

    def _cfg_to_qconfig(self, tune_cfg):
        """Convert the strategy config to MXNet quantization config.

        Args:
            tune_cfg (dict): tune config from lpot strategy.
                            cfg should be a format like below:
                            {
                                'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']],
                                'fp32': [['CONV2D', 'RELU', 'BN']]}, 'calib_iteration': 10,
                                'op': {
                                ['op1', 'CONV2D']: {
                                    'activation':  {'dtype': 'uint8',
                                                    'algorithm': 'minmax',
                                                    'scheme':'sym',
                                                    'granularity': 'per_tensor'},
                                    'weight': {'dtype': 'int8',
                                               'algorithm': 'kl',
                                               'scheme':'asym',
                                               'granularity': 'per_channel'}
                                },
                                ['op2', 'RELU]: {
                                    'activation': {'dtype': 'int8',
                                                   'scheme': 'asym',
                                                   'granularity': 'per_tensor',
                                                   'algorithm': 'minmax'}
                                },
                                ['op3', 'CONV2D']: {
                                    'activation':  {'dtype': 'fp32'},
                                    'weight': {'dtype': 'fp32'}
                                },
                                ...
                                }
                            }

        """
        excluded_sym_names = []
        excluded_op_names = []
        calib_minmax_layers = []
        calib_kl_layers = []

        for _, op in enumerate(self.quantizable_ops):
            # get qdata type per op
            if tune_cfg['op'][(op["name"], op["type"])
                              ]['activation']['dtype'] == 'fp32':
                excluded_sym_names.append(op["name"])
                continue
            # get calib algorithm per op
            if tune_cfg['op'][(op["name"], op["type"])
                              ]['activation']['algorithm'] == 'minmax':
                calib_minmax_layers.append(op["name"] + "_output")
            elif tune_cfg['op'][(op["name"], op["type"])]['activation']['algorithm'] == 'kl':
                calib_kl_layers.append(op["name"] + "_output")

        LayerOutputCollector = None
        # for not tunable config
        quantized_dtype = 'auto'
        quantize_mode = 'smart'
        quantize_granularity = 'tensor-wise'
        logger = self.logger
        ctx = mx.cpu()
        batch_size = self.__config_dict['calib_data'].batch_size
        iteration = tune_cfg['calib_iteration']
        num_calib_examples = batch_size * iteration

        self.__config_dict.update({
            "excluded_sym_names": excluded_sym_names,
            "excluded_op_names": excluded_op_names,
            "LayerOutputCollector": LayerOutputCollector,
            "quantized_dtype": quantized_dtype,
            "quantize_mode": quantize_mode,
            "quantize_granularity": quantize_granularity,
            "logger": logger,
            "ctx": ctx,
            "num_calib_examples": num_calib_examples,
            "iteration": iteration,
            "exclude_layers_match": [],
            "calib_kl_layers": calib_kl_layers,
            "calib_minmax_layers": calib_minmax_layers,
        })
        self.logger.debug('tuning configs of python API:\n %s, '
                          % (self.__config_dict))

    def _get_gluon_symbol(self, network, dataloader):
        """Convert symbol model and DataIter from gluon model HybridBlock/Dataloader.

        Args:
            network (HybridBlock): gluon HybridBlock model.
            dataloader (Dataloader): gluon Dataloader.

        Returns:
            tuple: symbol model and DataIter.
        """
        class _DataIterWrapper(mx.io.DataIter):
            """DataIter wrapper for general iterator, e.g., gluon dataloader"""

            def __init__(self, calib_data):
                self._data = calib_data
                try:
                    calib_iter = iter(calib_data)
                except TypeError as e:
                    raise TypeError(
                        'calib_data is not a valid iterator. {}'.format(
                            str(e)))
                data_example = next(calib_iter)
                if isinstance(data_example, (list, tuple)):
                    data_example = list(data_example)
                else:
                    data_example = [data_example]

                num_input = len(data_example)
                assert num_input > 0
                self.provide_data = [
                    mx.io.DataDesc(
                        name='data', shape=(
                            data_example[0].shape))]
                # data0, data1, ..., label
                if num_input >= 3:
                    self.provide_data = [mx.io.DataDesc(name='data{}'.format(i), shape=x.shape)
                                         for i, x in enumerate(data_example[0:-1])]
                self.batch_size = data_example[0].shape[0]
                self.reset()

            def reset(self):
                self._iter = iter(self._data)

            def next(self):
                next_data = next(self._iter)
                return mx.io.DataBatch(data=next_data)

        network.hybridize()
        calib_data = dataloader

        if calib_data is not None:
            if isinstance(calib_data, mx.io.DataIter):
                data_info = calib_data.provide_data
            else:
                calib_data = _DataIterWrapper(calib_data)
                data_info = calib_data.provide_data

        if not data_info:
            raise ValueError('need provide_data to infer data shape.')
        data_nd = []
        for shape in data_info:
            data_nd.append(mx.nd.zeros(shape.shape))
        while True:
            try:
                network(*data_nd)
            except TypeError:
                del data_nd[-1]
                del calib_data.provide_data[-1]
                continue
            else:
                break

        with TemporaryDirectory() as tmpdirname:
            prefix = os.path.join(tmpdirname, 'tmp')
            network.export(prefix, epoch=0)
            symnet, args, auxs = mx.model.load_checkpoint(prefix, 0)

        self.__config_dict['calib_data'] = calib_data

        return symnet, args, auxs, calib_data

    @dump_elapsed_time("Compute quantization scaling using KL algorithm")
    def _get_optimal_thresholds(self, hist_dict, quantized_dtype,
                                num_quantized_bins=255, logger=None):
        """Given a ndarray dict, find the optimal threshold for quantizing each value of the key.

        Args:
            hist_dict (dict): dict of each layer output tensor, format as {layer_name: tensor}.
            quantized_dtype (str): quantized data type.
            num_quantized_bins (int, optional): num_quantized_bins. Defaults to 255.
            logger (logger, optional): logger. Defaults to None.

        Returns:
            th_dict (dict): optimal_thresholds of each layer tensor.
        """
        assert isinstance(hist_dict, dict)
        self.logger.info(
            'Calculating optimal thresholds for quantization using KL divergence'
            ' with num_quantized_bins=%d' % num_quantized_bins)
        th_dict = {}
        # copy hist_dict keys since the keys() only returns a view in python3
        layer_names = list(hist_dict.keys())
        _kl = KL_Divergence()
        for name in layer_names:
            assert name in hist_dict
            (hist, hist_edges, min_val, max_val, _) = hist_dict[name]
            th = _kl.get_threshold(hist,
                                   hist_edges,
                                   min_val,
                                   max_val,
                                   num_bins=8001,
                                   quantized_type=quantized_dtype,
                                   num_quantized_bins=255)

            if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
                th_dict[name] = (0, th)
            else:
                th_dict[name] = (-th, th)
            del hist_dict[name]  # release the memory
            self.logger.debug('layer=%s, min_val=%f, max_val=%f, th=%f, '
                              % (name, min_val, max_val, th))
        return th_dict

    @dump_elapsed_time("Compute quantization scaling using minmax algorithm")
    def _get_min_max_thresholds(self, mod, calib_data, quantized_dtype,
                                include_layer, max_num_examples, logger):
        th_dict_min_max, num_examples = mx.contrib.quantization._collect_layer_output_min_max(
            mod, calib_data, quantized_dtype, include_layer=include_layer,
            max_num_examples=max_num_examples, logger=logger)

        logger.info('Collected layer output min/max values from FP32 model using %d examples' %
                    num_examples)
        return th_dict_min_max

    def _get_calibration_th(
            self,
            sym,
            arg_params,
            aux_params,
            calib_layer,
            qconfig):
        """Calculate the calibration value of each layer. the calibration method can be min/max
           or KL on different layers.

        Args:
            sym (object): model symbol file.
            arg_params (object): arg_params.
            aux_params (object): aux_params.
            calib_layer (list): layers need to do calibration.
            qconfig (dict): quantization config.

        Returns:
            th_dict (dict): dict include the calibration value of each layer.
        """
        ctx = qconfig['ctx']
        calib_data = qconfig['calib_data']
        num_calib_examples = qconfig['num_calib_examples']
        quantized_dtype = qconfig['quantized_dtype']
        logger = self.logger

        calib_data.reset()
        th_dict = {}

        if not isinstance(ctx, mx.context.Context):
            raise ValueError(
                'currently only supports single ctx, while received %s' %
                str(ctx))
        if calib_data is None:
            raise ValueError(
                'calib_data must be provided when doing calibration!')
        if not isinstance(calib_data, mx.io.DataIter):
            raise ValueError('calib_data must be of DataIter type ,'
                             ' while received type %s' % (str(type(calib_data))))

        data_names = [pair[0] for pair in calib_data.provide_data]
        # label_names = [pair[0] for pair in calib_data.provide_label]

        mod = mx.module.module.Module(
            symbol=sym, data_names=data_names, context=ctx)
        if hasattr(calib_data,
                   'provide_label') and len(
                       calib_data.provide_label) > 0:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data,
                     label_shapes=calib_data.provide_label)
        else:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data)
        mod.set_params(arg_params, aux_params)

        for data_name in data_names:
            # the data_name of gluon model may diff with the name of input data layer,
            # it caused by gluon model convert to symbol model
            if data_name in calib_layer:
                self.__config_dict["calib_minmax_layers"].append(data_name)

        if len(self.__config_dict["calib_kl_layers"]) != 0:
            # inspect each quantized layer activate tensor for calibration
            layer_tensor = self._inspect_tensor((sym, arg_params, aux_params),
                                                dataloader=calib_data,
                                                op_list=calib_layer)
            _histogram = LayerHistogramCollector(
                layer_tensor=layer_tensor,
                include_layer=self.__config_dict["calib_kl_layers"])
            _histogram.collect()
            hist_dict = _histogram.hist_dict
            self.logger.info('Calculating optimal thresholds for quantization')

            th_dict_kl = self._get_optimal_thresholds(
                hist_dict, quantized_dtype, logger=logger)
            self._merge_dicts(th_dict_kl, th_dict)

        calib_data.reset()
        if len(self.__config_dict["calib_minmax_layers"]) != 0:
            th_dict_minmax = self._get_min_max_thresholds(
                mod, calib_data, quantized_dtype,
                include_layer=self.__config_dict["calib_minmax_layers"],
                max_num_examples=num_calib_examples,
                logger=logger)
            self._merge_dicts(th_dict_minmax, th_dict)

        return th_dict

    def _merge_dicts(self, src, dst):
        """Merge src dict to dst dict

        Args:
            src (dict): source dict.
            dst (dict): dest dict.

        Returns:
            dst (dict): merged dict.
        """
        for key in src:
            assert key not in dst, "%s layer can not do KL and minmax calibration together!" % key
            if not isinstance(src[key], dict):
                dst[key] = src[key]

        return dst

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
                self.logger.info("Failed to parse {} due to {}".format(self.cfg, str(e)))
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
        """Get the current backend's version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self):
        """Get the supported low precisions, e.g ['int8', 'bf16']
        """
        return self.cur_config['precisions']['names']

    def get_op_types(self):
        """Get the op types for specific backend per low precision.
            e.g {'1.6.0': {'int8': ['Conv2D', 'fully_connected']}}
        """
        return self.cur_config['ops']

    def get_fuse_patterns(self):
        """Get the fusion patterns for specified op type for every specific precision

        """
        return self.cur_config['patterns']

    def get_quantization_capability(self):
        """Get the quantization capability of low precision op types.
            e.g, granularity, scheme and etc.

        """
        return self.cur_config['capabilities']

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
