from .adaptor import adaptor_registry, Adaptor
from ..utils import LazyImport
from ..algo.kl_divergence import KL_Divergence
from ..algo.collect_layer_histogram import LayerHistogramCollector
from collections import OrderedDict
import numpy as np

import os
import time
import logging
import shutil
import tempfile
import json
from tempfile import TemporaryDirectory

mx = LazyImport("mxnet")

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("iLit-MXNet")

@adaptor_registry
class MxNetAdaptor(Adaptor):
    def __init__(self, input_output_info):
        super(MxNetAdaptor, self).__init__(input_output_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.logger = logger
        self.qdataloader = input_output_info["q_dataloader"]

    def _get_backedn_graph(self, symbol, ctx):
        if ctx == mx.cpu():
            return symbol.get_backend_symbol('MKLDNN_QUANTIZE')
        else:
            pass

    def quantize(self, tune_cfg, model, dataloader):
        '''The function is used to do calibration and quanitization in post-training quantization.
           It is used to do quanization in quantization-aware training.

           Args:
               model (object): The model to do calibration.
        '''
        self.cfg = tune_cfg
        self._cfg_to_qconfig(tune_cfg)
        self.__config_dict['calib_data'] = dataloader
        self.th_dict = None
        qconfig = self.__config_dict
        logger.info(qconfig)
        # print mxnet quantization verbose for debug
        # os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'

        # get symbol from FP32 model
        if isinstance(model, mx.gluon.HybridBlock):
            # transfer hybridblock to symbo
            sym, arg_params, aux_params, calib_data = self._get_gluon_symbol(model, dataloader=dataloader)
            data_names = [pair[0] for pair in calib_data.provide_data]
        elif isinstance(model[0], mx.symbol.Symbol):
            sym, arg_params, aux_params = model
        else:
            raise ValueError(
                'Need a symbol model or HybridBlock model, while received %s' % str(type(model)))
        sym = self._get_backedn_graph(sym, qconfig['ctx'])

        # 1. quantize_symbol
        qsym, calib_layer = mx.contrib.quantization._quantize_symbol(sym, qconfig['ctx'], excluded_symbols=qconfig['excluded_sym_names'],
                                            excluded_operators=qconfig['excluded_op_names'],
                                            offline_params=list(arg_params.keys()),
                                            quantized_dtype=qconfig['quantized_dtype'],
                                            quantize_mode=qconfig['quantize_mode'],
                                            quantize_granularity=qconfig['quantize_granularity'])
        # 2. Do calibration to get th_dict
        th_dict = self._get_calibration_th(sym, arg_params, aux_params, calib_layer, qconfig)
        self.th_dict = th_dict
        # 3. set th_dict to quantized symbol
        qsym = mx.contrib.quantization._calibrate_quantized_sym(qsym, th_dict)
        # 4. quantize params
        qarg_params = mx.contrib.quantization._quantize_params(qsym, arg_params, th_dict)
        qsym = self._get_backedn_graph(qsym, qconfig['ctx'])

        if isinstance(model, mx.gluon.HybridBlock):
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
                mx.ndarray.save(param_name, save_dict)
                net.collect_params().load(param_name, cast_dtype=True, dtype_source='saved')
                net.collect_params().reset_ctx(self.__config_dict['ctx'])
                return net

        return (qsym, qarg_params, aux_params)

    def train(self, model, dataloader):
        '''The function is used to do training in quantization-aware training.

           Args:
               model (object): The model to do calibration.
        '''
        raise notimplementederror

    def evaluate(self, model, dataloader, metric):
        '''The function is used to run evaluation on validation dataset.

           Args:
               model (object): The model to do calibration.
        '''

        if isinstance(model, mx.gluon.HybridBlock):
            acc = self._mxnet_gluon_forward(model, dataloader, metric)

        elif isinstance(model[0], mx.symbol.symbol.Symbol):
            assert isinstance(dataloader, mx.io.DataIter), \
                    'need mx.io.DataIter. but recived %s' % str(type(dataloader))
            dataloader.reset()
            acc = self._mxnet_symbol_forward(model, dataloader, metric)

        else:
            raise ValueError("Unknow graph tyep: %s" %(str(type(model))))

        return acc

    def _mxnet_symbol_forward(self, symbol_file, dataIter, metric):

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
        for batch in dataIter:
            mod.forward(batch, is_train=False)
            output = mod.get_outputs()
            output = output[0].asnumpy()
            label = batch.label[0].asnumpy()
            acc = metric.evaluate(output, label)
            batch_num += dataIter.batch_size
            # for test, only forward 2 iters
            # if batch_num >= 2:
            #     break

        return acc

    def _mxnet_gluon_forward(self, gluon_model, dataloader, metrics):

        data_l, label_l = pre_process(dataloader)
        metric = metrics[0]
        metric.reset()
        batch_num = 0
        # pdb.set_trace()
        for data, label in zip(data_l, label_l):
            out = gluon_model(*data)
            metric.update(label, out)
            batch_num += len(data_l[0][0])
        res = metric.get()
        if len(res) == 1:
            acc = res[1]

        else:
            acc = res[1][0]

        return acc

    def _check_model(self, model, dataloader):
        '''The function is used to check model and calib_data, if not symbol and dataiter, then transfer it to.

           Args:
                model (object): The model to do calibration.
                config:
           Return:
                sym, arg_params, aux_params
        '''
        if isinstance(model, mx.gluon.HybridBlock):
            # model.hybridblock()
            sym, arg_params, aux_params, calib_data = self._get_gluon_symbol(network=model, dataloader=dataloader)
            self.__config_dict['calib_data'] = calib_data
        elif isinstance(model[0], mx.symbol.Symbol):
            sym, arg_params, aux_params = model
            calib_data = dataloader
        else:
            raise TypeError

        return sym, arg_params, aux_params, calib_data

    def _query_quantizable_ops(self, model, calib_data):
        '''The function is used to run test on validation dataset.

           Args:
               model (object): The model to do calibration.
        '''
        if len(self.quantizable_ops) != 0:
            return self.quantizable_ops

        sym, arg_params, aux_params, calib_data = self._check_model(model, calib_data)
        sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

        _, calib_layer = mx.contrib.quantization._quantize_symbol(sym, mx.cpu(), offline_params=list(arg_params.keys()),)

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
                symbol_layers.append({"name": name, "type":type})

        for _, opname_type in enumerate(symbol_layers):
            if opname_type["name"] + "_output" in calib_layer:
                self.quantizable_ops.append(opname_type)

        return self.quantizable_ops

    def query_fused_patterns(self, model):
        '''The function is used to run fused patterns in framework.

           Args:
               model (object): The model to do calibration.
        '''
        raise notimplementederror

    def query_fw_capability(self, model):
        # model_wise capability
        # TODO: weight granularity
        model_wise = {
                'activation': {'data_type': ['uint8', 'fp32'], \
                    'granularity': ['per_channel'], 'algo': ['minmax', 'kl']},
                'weight': {'data_type': ['uint8', 'fp32'], \
                    'granularity': ['per_channel'], 'algo': ['minmax', 'kl']}
                }
        # op_wise capability
        quantizable_ops = self._query_quantizable_ops(model, self.qdataloader)
        op_wise = OrderedDict()
        for _, opname_type in enumerate(quantizable_ops):
            optype = opname_type["type"]
            if optype in ['_sg_mkldnn_conv','conv2d']:
                op_capability = {
                    'activation': {
                        'data_type': ['uint8', 'fp32'],
                        'algo': ['minmax', 'kl'],
                        'granularity': ['per_channel']},
                    'weight': {
                        'data_type':['uint8', 'fp32'],
                        'granularity': ['per_channel']}
                    }
            elif optype in ['_sg_fully_connected','fully_connected']:
                op_capability = {
                    'activation': {
                        'data_type': ['uint8', 'fp32'],
                        'algo': ['minmax', 'kl'],
                        'granularity': ['per_channel']},

                    'weight': {
                        'data_type':['uint8', 'fp32'],
                        'granularity': ['per_channel']}
                    }
            elif optype in ['relu']:
                op_capability = {
                    'activation': {
                        'data_type': ['uint8', 'fp32'],
                        'algo': ['minmax', 'kl'],
                        'granularity': ['per_tensor']}
                    }
            else:
                op_capability = {
                    'activation': {
                            'data_type': ['uint8', 'fp32'],
                            'algo': ['minmax', 'kl'],
                            'granularity': ['per_channel']},
                    'weight': {
                            'data_type':['uint8', 'fp32'],
                            'granularity': ['per_channel']}
                    }

            op_wise.update({(opname_type["name"], opname_type["type"]): op_capability})

        return {'modelwise': model_wise, 'opwise': op_wise}

    def _inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        '''The function is used by tune strategy class for dumping tensor info.

           Args:
               model (object): The model to do calibration.
               dataloader: The data to do forword.
               op_list: list of inspect tensors.
               iteration_list: list of inspect iterations
           Return:
               Numpy Array Dict
                if iteration_list is empty:
                    {'op1':  tensor, ...}
                if iteration_list is not empty:
                    {"iteration_1": {'op1': tensor, ...}
                     "iteration_2": {'op1': tensor, ...}}
        '''
        import ctypes
        from mxnet.base import _LIB, check_call, py_str
        from mxnet.base import c_array, c_str, mx_uint, c_str_array
        from mxnet.base import NDArrayHandle, SymbolHandle
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
                arr = mx.ndarray.NDArray(handle, writable=False)
                if name in self.tensor_dict.keys():
                    self.tensor_dict[name].append(arr.asnumpy())
                else:
                    self.tensor_dict[name] = [arr.asnumpy()]

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
        mod = mx.module.module.Module(symbol=sym, data_names=data_names, context=self.__config_dict['ctx'])
        mod.bind(for_training=False, data_shapes=data.provide_data)
        mod.set_params(arg_params, aux_params)
        mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
        num_batches = 0

        if len(iteration_list) == 0:
            for _, batch in enumerate(data):
                mod.forward(data_batch=batch, is_train=False)
                num_batches += 1
                if calib_iter is not None and num_batches >= calib_iter:
                    break
            if logger is not None:
                logger.info("Inspect tensors from %d batches with batch_size=%d"
                            % (num_batches, data.batch_size))

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

            if logger is not None:
                logger.info("Inspect tensors from %d batches with batch_size=%d"
                            % (num_batches, data.batch_size))
            return iter_tensor

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        int8_ops_th = self.th_dict
        op_list_convert = []
        sym = model[0]
        sym_all_layers = [layer.name for layer in list(sym.get_internals())]
        for item in op_list:
            if "quantized_" + item in sym_all_layers:
                item = "quantized_" + item
                int8_ops_th
            if not item.endswith("_output"):
                item += "_output"
            op_list_convert.append(item)
        
        inspected_tensor = self._inspect_tensor(model, dataloader, op_list_convert, iteration_list)
        inspected_tensor_convert = {}
        for op, tensor in inspected_tensor.items():
            if op.startswith("quantized_"):
                op = op[10:]
                if op in int8_ops_th:
                    op_min = mx.nd.array(int8_ops_th[op][0])
                    op_max = mx.nd.array(int8_ops_th[op][1])
                    #TODO: deal hard code dtype
                    tensor = mx.nd.contrib.dequantize(mx.nd.array(tensor, dtype='uint8'), \
                        min_range=op_min, max_range=op_max, out_type='float32').asnumpy()
                    assert tensor.dtype == np.float32
            if op.endswith("_output"):
                op = op[:-7]

            inspected_tensor_convert.update({op: tensor})

        return inspected_tensor_convert

    def mapping(self, src_model, dst_model):
        '''The function is used to create a dict to map tensor name of src model to tensor name of dst model.

           Return:
               Dict
               {'src_op1': 'dst_op1'}
        '''
        raise notimplementederror

    def save(self, model):
        '''The function is used by tune strategy class for saving model.

           Args:
               model (object): The model to do calibration.
        '''
        output_dir = './quantize_model/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_name = 'iLit' + '_quantized_model'
        params_saved = os.path.join(output_dir, ckpt_name)
        if isinstance(model, mx.gluon.HybridBlock):
            logger.info("Save MXNet HybridBlock quantization model!")
            model.export(params_saved, epoch=0)
            logging.info('Saving quantized model at %s', output_dir)
        else:
            logger.info('Saving symbol into file at %s' % ckpt_name)
            symbol, arg_params, aux_params = model
            symbol.save(params_saved+'-symbol.json')
            save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
            save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
            mx.nd.save(params_saved+'-0000.params', save_dict)

    def _cfg_to_qconfig(self, tune_cfg):

        '''cfg should be a format like below:
        {
            'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']], 'fp32': [['CONV2D', 'RELU', 'BN']]},
            'calib_iteration': 10,
            'op': {
               ['op1', 'CONV2D']: {
                 'activation':  {'data_type': 'uint8', 'algo': 'minmax', 'mode':'sym', 'granularity': 'per_tensor'},
                 'weight': {'data_type': 'int8', 'algo': 'kl', 'mode':'asym', 'granularity': 'per_channel'}
               },
               ['op2', 'RELU]: {
                 'activation': {'data_type': 'int8', 'mode': 'asym', 'granularity': 'per_tensor', 'algo': 'minmax'}
               },
               ['op3', 'CONV2D']: {
                 'activation':  {'data_type': 'fp32'},
                 'weight': {'data_type': 'fp32'}
               },
               ...
            }
          }

        '''
        excluded_sym_names = []
        excluded_op_names = []
        calib_minmax_layers = []
        calib_kl_layers = []

        for _, op in enumerate(self.quantizable_ops):
            # get qdata type per op
            if tune_cfg['op'][(op["name"], op["type"])]['activation']['data_type'] == 'fp32':
                excluded_sym_names.append(op["name"])
                continue
            # get calib algo per op
            if tune_cfg['op'][(op["name"], op["type"])]['activation']['algo'] == 'minmax':
                calib_minmax_layers.append(op["name"] + "_output")
            elif tune_cfg['op'][(op["name"], op["type"])]['activation']['algo'] == 'kl':
                calib_kl_layers.append(op["name"] + "_output")

        LayerOutputCollector = None
        # for not tunable config
        quantized_dtype = 'auto'
        quantize_mode = 'smart'
        quantize_granularity = 'channel-wise'
        logger = None
        ctx = mx.cpu()
        calib_data = None
        batch_size = 64
        iteration = 2
        if 'calib_iteration' in list(tune_cfg.keys()):
            iteration = tune_cfg['calib_iteration']
        num_calib_examples = batch_size * iteration

        self.__config_dict = {
            "excluded_sym_names" : excluded_sym_names,
            "excluded_op_names" : excluded_op_names,
            "LayerOutputCollector" : LayerOutputCollector,
            "quantized_dtype" : quantized_dtype,
            "quantize_mode" : quantize_mode,
            "quantize_granularity" : quantize_granularity,
            "logger" : logger,
            "ctx" : ctx,
            "calib_data": calib_data,
            "num_calib_examples":num_calib_examples,
            "iteration":iteration,
            # "label_names":Label_names,
            # "data_names":Data_names,
            "exclude_layers_match":[],
            "calib_kl_layers": calib_kl_layers,
            "calib_minmax_layers": calib_minmax_layers,
        }

    def _get_gluon_symbol(self, network, dataloader):

        network.hybridize()
        calib_data = dataloader

        if calib_data is not None:
            if isinstance(calib_data, mx.io.DataIter):
                dshapes = calib_data.provide_data
            else:
                calib_data, dshapes = mx.contrib.quantization._as_data_iter(calib_data)
        data_shapes = dshapes

        if not data_shapes:
            raise ValueError('data_shapes required')
        data_nd = []
        for shape in data_shapes:
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

    def _get_optimal_thresholds(self, hist_dict, quantized_dtype, num_quantized_bins=255, logger=None):
        """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
        assert isinstance(hist_dict, dict)
        if logger is not None:
            logger.info('Calculating optimal thresholds for quantization using KL divergence'
                        ' with num_quantized_bins=%d' % num_quantized_bins)
        th_dict = {}
        # copy hist_dict keys since the keys() only returns a view in python3
        layer_names = list(hist_dict.keys())
        ilit_kl = KL_Divergence()
        for name in layer_names:
            assert name in hist_dict
            # min_val, max_val, th, divergence = \
            #     _get_optimal_threshold(hist_dict[name], quantized_dtype,
            #                            num_quantized_bins=num_quantized_bins)
            (hist, hist_edges, min_val, max_val, _) = hist_dict[name]
            th = ilit_kl.get_threshold(hist,
                                        hist_edges,
                                        min_val,
                                        max_val,
                                        num_bins = 8001,
                                        quantized_type=quantized_dtype,
                                        num_quantized_bins=255)

            if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
                th_dict[name] = (0, th)
            else:
                th_dict[name] = (-th, th)
            del hist_dict[name]  # release the memory
            if logger:

                logger.debug('layer=%s, min_val=%f, max_val=%f, th=%f, '
                            % (name, min_val, max_val, th))
        return th_dict

    def _get_calibration_th(self, sym, arg_params, aux_params, calib_layer, qconfig):
        # calib_mode = qconfig['calib_mode']
        ctx = qconfig['ctx']
        calib_data = qconfig['calib_data']
        num_calib_examples = qconfig['num_calib_examples']
        quantized_dtype = qconfig['quantized_dtype']
        logger = self.logger

        calib_data.reset()
        th_dict = {}

        if not isinstance(ctx, mx.context.Context):
            raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
        if calib_data is None:
            raise ValueError('calib_data must be provided when doing calibration!')
        if not isinstance(calib_data, mx.io.DataIter):
            raise ValueError('calib_data must be of DataIter type ,'
                            ' while received type %s' % (str(type(calib_data))))

        data_names = [pair[0] for pair in calib_data.provide_data]
        # label_names = [pair[0] for pair in calib_data.provide_label]

        mod = mx.module.module.Module(symbol=sym, data_names=data_names, context=ctx)
        # mod = Module(symbol=sym)
        if hasattr(calib_data, 'provide_label') and len(calib_data.provide_label) > 0:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data,
                    label_shapes=calib_data.provide_label)
        else:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data)
        mod.set_params(arg_params, aux_params)

        # inspect each quantized layer activate tensor for calibration
        layer_tensor = self._inspect_tensor((sym, arg_params, aux_params), dataloader=calib_data, op_list=calib_layer)

        if len(self.__config_dict["calib_kl_layers"]) != 0:
            iLiT_histogram = LayerHistogramCollector(layer_tensor=layer_tensor, include_layer=self.__config_dict["calib_kl_layers"])
            iLiT_histogram.collect()
            hist_dict = iLiT_histogram.hist_dict
            if logger:
                logger.info('Calculating optimal thresholds for quantization')
            th_dict_kl = self._get_optimal_thresholds(hist_dict, quantized_dtype, logger=logger)
            self._merge_dicts(th_dict_kl, th_dict)
            if logger:
                logger.info('Collected layer output KL values from FP32 model')

        if len(self.__config_dict["calib_minmax_layers"]) != 0:
            th_dict_minmax, num_examples = mx.contrib.quantization._collect_layer_output_min_max(
                mod, calib_data, quantized_dtype, include_layer=self.__config_dict["calib_minmax_layers"], max_num_examples=num_calib_examples,
                logger=logger)
            self._merge_dicts(th_dict_minmax, th_dict)
            if logger:
                logger.info('Collected layer output min/max values from FP32 model using %d examples'
                            % num_examples)

        return th_dict

    def _merge_dicts(self, src, dst):
        '''Merges src calib th dict into dst calib th dict'''
        for key in src:
            assert key not in dst, "%s layer can not do KL and minmax calibration together!" % key
            if not isinstance(src[key], dict):
                dst[key] = src[key]

        return dst
