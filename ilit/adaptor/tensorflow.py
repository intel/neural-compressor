import os
import subprocess
import copy
import time
import numpy as np
from collections import OrderedDict
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport
from ..utils import logger
tensorflow = LazyImport('tensorflow')


@adaptor_registry
class TensorFlowAdaptor(Adaptor):
    unify_op_type_mapping = {
        "Conv2D": "conv2d",
        "DepthwiseConv2dNative": "conv2d",
        "MaxPool": "pooling",
        "AvgPool": "pooling",
        "ConcatV2": "concat",
        "MatMul": "matmul",
        "Pad": "pad"
    }

    def __init__(self, framework_specific_info):
        super(TensorFlowAdaptor, self).__init__(framework_specific_info)

        self.quantize_config = {'op_wise_config': {}}
        self.framework_specific_info = framework_specific_info
        self.inputs = self.framework_specific_info['inputs']
        self.outputs = self.framework_specific_info['outputs']
        self.device = self.framework_specific_info['device']
        self.bf16_ops = []
        self.fp32_ops = []

    def get_tensor_by_name_with_import(self, graph, name, try_cnt=3):
        """Get the tensor by name considering the 'import' scope when model
           may be imported more then once

        Args:
            graph (tf.compat.v1.GraphDef): the model to get name from 
            name (string): tensor name do not have 'import'

        Returns:
            tensor: evaluation result, the larger is better.
        """
        for i in range(0, try_cnt):
            try:
                return graph.get_tensor_by_name(name)
            except:
                name = 'import/' + name
        raise ValueError('can not find tensor by name')

    def evaluate(self, input_graph, dataloader, postprocess=None, \
                 metric=None, measurer=None, iteration=-1):
        """Evaluate the model for specified metric on validation dataset.

        Args:
            input_graph ([Graph, GraphDef or Path String]): The model could be the graph,
                          graph_def object, the frozen pb or ckpt/savedmodel folder path.
            dataloader (generator): generate the data and labels.
            metric (object, optional): Depends on model category. Defaults to None.
            measurer (object, optional): for precise benchmark measurement.

        Returns:
            [float]: evaluation result, the larger is better.
        """
        logger.info("start to evaluate model....")
        from .tf_utils.util import get_graph_def
        import tensorflow as tf
        graph = tf.Graph()
        graph_def = get_graph_def(input_graph)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference 
        from tensorflow.python.framework import dtypes
        graph_def = optimize_for_inference(graph_def, [self.inputs[0]], self.outputs, \
                                           dtypes.float32.as_datatype_enum, False) 
        assert graph_def
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        input_tensor = self.get_tensor_by_name_with_import(graph, self.inputs[0] + ":0")
        output_tensor = [
            self.get_tensor_by_name_with_import(graph, x + ":0") for x in self.outputs
        ]

        config = tf.compat.v1.ConfigProto() 
        config.use_per_session_threads = 1
        # config.intra_op_parallelism_threads = 28
        config.inter_op_parallelism_threads = 1

        logger.info("Start to evaluate model via tensorflow...")

        sess_graph = tf.compat.v1.Session(graph=graph, config=config)

        for idx, (images, labels) in enumerate(dataloader):
            if measurer is not None:
                measurer.start()
                predictions = sess_graph.run(output_tensor, {input_tensor: images})
                measurer.end()
            else:
                predictions = sess_graph.run(output_tensor, {input_tensor: images})
            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metric is not None:
                metric.update(predictions[0], labels)
            if idx + 1 == iteration:
                break
        acc = metric.result() if metric is not None else 0
        sess_graph.close()
        return acc

    def tuning_cfg_to_fw(self, tuning_cfg):
        """Parse the ilit wrapped configuration to Tensorflow.

        Args:
            tuning_cfg (dict): configuration for quantization.
        """
        self.quantize_config['calib_iteration'] = tuning_cfg['calib_iteration']
        self.quantize_config['device'] = self.device
        fp32_ops = []
        bf16_ops = []
        for each_op_info in tuning_cfg['op']:
            op_name = each_op_info[0]

            if tuning_cfg['op'][each_op_info]['activation']['dtype'] in ['fp32', 'bf16']:
                if op_name in self.quantize_config['op_wise_config']:
                    self.quantize_config['op_wise_config'].pop(op_name)
                if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'fp32':
                    fp32_ops.append(op_name)
                if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'bf16':
                    bf16_ops.append(op_name)
                continue

            is_perchannel = False
            if 'weight' in tuning_cfg['op'][each_op_info]:
                is_perchannel = tuning_cfg['op'][each_op_info]['weight'][
                    'granularity'] == 'per_channel'
            algorithm = tuning_cfg['op'][each_op_info]['activation'][
                'algorithm']

            is_asymmetric = False
            if 'activation' in tuning_cfg['op'][each_op_info]:
                is_asymmetric = tuning_cfg['op'][each_op_info]['activation'][
                    'scheme'] == 'asym'
            self.quantize_config['op_wise_config'][op_name] = (is_perchannel,
                                                               algorithm, is_asymmetric)
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        int8_sum_count = 0
        bf16_sum_count = 0
        log_length = 50
        print ('|', 'Mixed Precision Statistics'.center(log_length, "*"), "|")
        for i in self._init_op_stat:
            if len(self._init_op_stat[i]) == 0:
                continue
            count = 0
            for j in self.quantize_config['op_wise_config'].keys():
                if j in self._init_op_stat[i]:
                    count += 1
            int8_sum_count += count
            print ('|', 'INT8 {}: {} '.format(i, count).ljust(log_length), "|")
            bf16_count = 0
            for k in self.bf16_ops:
                if k in self._init_op_stat[i]:
                    bf16_count += 1
                if bf16_count > 0:
                    print ('|', 'BF16 {}: {}'.format(i, bf16_count).ljust(log_length), "|")
            bf16_sum_count += bf16_count
        overall_ops_count = sum([len(v) for _, v in self._init_op_stat.items()])
        int8_percent = float(int8_sum_count/overall_ops_count)
        bf16_percent = float(bf16_sum_count/overall_ops_count)
        print('|', 'Overall: INT8 {:.2%} ({}/{}) BF16 {:.2%} ({}/{})'.format(int8_percent,
                                                                            int8_sum_count,
                                                                            overall_ops_count,
                                                                            bf16_percent,
                                                                            bf16_sum_count,
                                                                            overall_ops_count)
                                                                            .ljust(log_length),
                                                                            "|")
        print('|', '*'*log_length, "|")

    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            data_loader (generator): generator the data and labels
            q_func (optional): training function for quantization aware training mode,
                               unimplement yet for tensorflow

        Returns:
            tf.compat.v1.GraphDef: the quantized model
        """
        assert q_func is None, "quantization aware training mode is not support on tensorflow"
        logger.info('Start to run model quantization...')
        quantized_model = os.path.join(os.getcwd(), "tf_quantized.pb")
        self.tuning_cfg_to_fw(tune_cfg)
        logger.debug('Dump quantization configurations:')
        logger.debug(self.quantize_config)
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   quantized_model,
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   qt_config=self.quantize_config,
                                   fp32_ops=self.fp32_ops,
                                   bf16_ops=self.bf16_ops,
                                   data_loader=data_loader)
        return converter.convert()

    def _query_quantizable_ops(self, graph, activation_dtype, weight_dtype):
        """Collect the op-wise configuration for quantization.

        Returns:
            OrderDict: op-wise configuration.
        """
        from .tf_utils.util import get_graph_def

        graph_def = get_graph_def(graph)
        assert graph_def
        tf_quantizable_op_type = ("Conv2D", "DepthwiseConv2dNative", "MaxPool",
                                  "AvgPool", "ConcatV2", "MatMul", "Pad")
        conv_config = {
            'activation': {
                'dtype': activation_dtype,
                'algorithm': ['minmax', 'kl'],
                'scheme': ['sym'],
                'granularity': ['per_tensor']
            },
            'weight': {
                'dtype': weight_dtype,
                'algorithm': ['minmax'],
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor']
            }
        }
        matmul_config = {
            'activation': {
                'dtype': activation_dtype,
                'algorithm': ['minmax'],
                'scheme': ['asym', 'sym'],
                'granularity': ['per_tensor']
            },
            'weight': {
                'dtype': weight_dtype,
                'algorithm': ['minmax'],
                'scheme': ['sym'],
                'granularity': ['per_tensor']
            }
        }
        other_config = {
            'activation': {
                'dtype': activation_dtype,
                'algorithm': ['minmax'],
                'scheme': ['sym'],
                'granularity': ['per_tensor']
            },
        }

        self.quantizable_op_details = OrderedDict()
        self._init_op_stat = {i:[] for i in tf_quantizable_op_type}
        for node in graph_def.node:
            if node.op in tf_quantizable_op_type:
                self._init_op_stat[node.op].append(node.name)

                if self.unify_op_type_mapping[node.op].find("conv2d") != -1:
                    self.quantizable_op_details[(
                        node.name, self.unify_op_type_mapping[node.op]
                    )] = copy.deepcopy(conv_config)
                elif self.unify_op_type_mapping[node.op].find("matmul") != -1:
                    self.quantizable_op_details[(
                        node.name, self.unify_op_type_mapping[node.op]
                    )] = copy.deepcopy(matmul_config)
                else:
                    self.quantizable_op_details[(
                        node.name, self.unify_op_type_mapping[node.op]
                    )] = copy.deepcopy(other_config)

                self.quantize_config['op_wise_config'][node.name] = (False,
                                                                     "minmax", False)
        return self.quantizable_op_details

    def _support_bf16(self):
        """Query Software and Hardware BF16 support cabability

        """
        import tensorflow as tf
        is_supported_version = False
        from tensorflow import python
        if (hasattr(python, "pywrap_tensorflow")
                and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):
            from tensorflow.python.pywrap_tensorflow import IsMklEnabled
        else:
            from tensorflow.python._pywrap_util_port import IsMklEnabled
        if IsMklEnabled() and (tf.version.VERSION >= "2.3.0"):
            is_supported_version = True
        command = "cat /proc/cpuinfo | grep flags | tail -n 1"
        all_flags = subprocess.check_output(command, shell=True).strip().decode()  # nosec
        if ((is_supported_version and " avx512_bf16 " in all_flags)
                or os.getenv('FORCE_BF16') == '1'):
            return True
        return False

    def query_fw_capability(self, model):
        """Collect the model-wise and op-wise configuration for quantization.

        Args:
            model (tf.compat.v1.GraphDef): model definition.

        Returns:
            [dict]: model-wise & op-wise configuration for quantization.
        """
        activation_dtype = ['uint8', 'fp32']
        weight_dtype = ['int8', 'fp32']
        if self._support_bf16():
            activation_dtype.append('bf16')
            weight_dtype.append('bf16')
        capability = {
            'modelwise': {
                'activation': {
                    'dtype': activation_dtype,
                    'scheme': ['asym', 'sym'],
                    'granularity': ['per_tensor'],
                    'algorithm': ['minmax']
                },
                'weight': {
                    'dtype': weight_dtype,
                    'scheme': [
                        'sym',
                    ],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax']
                },
            }
        }
        self._query_quantizable_ops(model, activation_dtype, weight_dtype)
        capability['opwise'] = self.quantizable_op_details
        logger.debug('Dump framework quantization capability:')
        logger.debug(capability)

        return capability

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        """Collect the specified tensor's output on specified iteration.

        Args:
            model (tf.compat.v1.GraphDef): model definition.
            dataloader (generator): generate the data and labels
            op_list (list, optional): the specified op names' list. Defaults to [].
            iteration_list (list, optional): the specified iteration. Defaults to [].

        Returns:
            [dict]: the key is op_name while the value is the ndarray tensor.
        """
        logger.info("Start to run inspect_tensor..")
        quantized_model = os.path.join(os.getcwd(), "tf_quantized.pb")
        from .tf_utils.graph_converter import GraphConverter

        converter = GraphConverter(model,
                                   quantized_model,
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   qt_config=self.quantize_config,
                                   data_loader=dataloader)
        return converter.inspect_tensor(op_list, iteration_list)
