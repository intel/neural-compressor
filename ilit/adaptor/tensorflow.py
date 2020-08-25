import os
import subprocess
import copy

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

    def evaluate(self, graph, dataloader, postprocess=None, metric=None):
        """Evaluate the model for specified metric on validation dataset.

        Args:
            graph (tf.compat.v1.GraphDef): the model for evaluate/
            dataloader (generator): generate the data and labels.
            metric (object, optional): Depends on model category. Defaults to None.

        Returns:
            [float]: evaluation result, the larger is better.
        """
        logger.info("start to evaluate model....")
        input_tensor = graph.get_tensor_by_name(self.inputs[0] + ":0")
        output_tensor = [
            graph.get_tensor_by_name(x + ":0") for x in self.outputs
        ]

        import tensorflow as tf

        num_inter_threads = 2
        num_intra_threads = int(
            subprocess.check_output(
                'cat /proc/cpuinfo | grep "cpu cores"|uniq|cut -d ":" -f 2',
                shell=True))

        config = tf.compat.v1.ConfigProto()
        config.inter_op_parallelism_threads = num_inter_threads
        config.intra_op_parallelism_threads = num_intra_threads

        sess_graph = tf.compat.v1.Session(graph=graph, config=config)
        logger.info("Start to evaluate model via tensorflow...")
        for images, labels in dataloader:
            predictions = sess_graph.run(output_tensor, {input_tensor: images})
            if metric is not None:
                metric.update(predictions[0], labels)
        acc = metric.result() if metric is not None else 0
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

    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            data_loader (generator): generator the data and labels
            q_func (optional): training function for quantization aware training mode, unimplement yet for tensorflow

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
                                   fp32_ops = self.fp32_ops,
                                   bf16_ops = self.bf16_ops,
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
        for node in graph_def.node:
            if node.op in tf_quantizable_op_type:
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
        all_flags = subprocess.check_output(command, shell=True).strip().decode()
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
