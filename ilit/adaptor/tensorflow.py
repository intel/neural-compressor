from .adaptor import adaptor_registry, Adaptor
from ..utils import LazyImport

from collections import OrderedDict
import os
import multiprocessing
import subprocess

tensorflow = LazyImport('tensorflow')

@adaptor_registry
class TensorFlowAdaptor(Adaptor):
    unify_op_type_mapping = {
            "Conv2D": "conv2d",
            "DepthwiseConv2dNative": "conv2d",
            "MaxPool": "pooling",
            "AvgPool": "pooling",
            "ConcatV2": "concat",
            "MatMul": "matmul"
        }

    def __init__(self, framework_specific_info):
        super(TensorFlowAdaptor, self).__init__(framework_specific_info)

        self.quantize_config = {}
        self.quantize_config['op_wise_config'] = {}
        self.framework_specific_info = framework_specific_info
        self.inputs = self.framework_specific_info['inputs'].split(',')
        self.outputs = self.framework_specific_info['outputs'].split(',')

    def evaluate(self, graph, dataloader, metric=None):
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
        print ("Start to evaluate model...")
        # batch = 0
        for content in dataloader:
            try:
                np_images, np_labels = content[0], content[1]

                predictions = sess_graph.run(output_tensor,
                                                {input_tensor: np_images})
                # print("Processed %d batches."% (batch + 1))
                # batch += 1
                acc = metric.evaluate(predictions[0], np_labels)

            except tf.errors.OutOfRangeError:
                print("Running out of images from dataset.")
                break
        return acc

    def tuning_cfg_to_fw(self, tuning_cfg):
        # TODO add the op-wise config parse
        self.excluded_nodes = []
        self.quantize_config['calib_iteration'] = tuning_cfg['calib_iteration']
        for each_op_info in tuning_cfg['op']:
            op_name = each_op_info[0]

            if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'fp32':
                self.excluded_nodes.append(op_name)
                continue
            is_perchannel = False
            if 'weight' in tuning_cfg['op'][each_op_info]:
                is_perchannel = tuning_cfg['op'][each_op_info]['weight']['granularity'] == 'per_channel'
            algorithm = tuning_cfg['op'][each_op_info]['activation']['algorithm']
            self.quantize_config['op_wise_config'][op_name] = (
                is_perchannel, algorithm)

    def quantize(self, tune_cfg, model, data_loader):
        quantized_model = os.path.join(os.getcwd(), "tf_quantized.pb")
        self.tuning_cfg_to_fw(tune_cfg)
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model, quantized_model,
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   qt_config=self.quantize_config,
                                   data_loader=data_loader)
        return converter.convert()

    def _query_quantizable_ops(self, graph):
        '''
            Return: Op name/Op type mapping which saved in OrderDict.
        '''
        graph_def = graph.as_graph_def()
        tf_quantizable_op_type = (
            "Conv2D", "DepthwiseConv2dNative", "MaxPool", "AvgPool", "ConcatV2", "MatMul")
        conv_config = {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'algorithm': ['minmax', 'kl'],
                'scheme': ['sym'],
                'granularity': ['per_tensor']
            },
            'weight': {
                'dtype': ['int8', 'fp32'],
                'algorithm': ['minmax'],
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor']
            }
        }
        non_conv_config = {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'algorithm': ['minmax'],
                'scheme': ['sym'],
                'granularity': ['per_tensor']
            },
        }

        self.quantizable_op_details = OrderedDict()
        for node in graph_def.node:
            if node.op in tf_quantizable_op_type:
                self.quantizable_op_details[(
                    node.name, self.unify_op_type_mapping[node.op]
                )] = conv_config if self.unify_op_type_mapping[node.op].find(
                    "conv2d") != -1 else non_conv_config
                self.quantize_config['op_wise_config'][node.name] = (
                    False, "minmax")

        return self.quantizable_op_details

    def query_fw_capability(self, model):
        capability = {
            'modelwise': {
                'activation': {
                    'dtype': ['uint8', 'fp32'],
                    'scheme': ['sym'],
                    'granularity': ['per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
                'weight': {
                    'dtype': ['int8',  'fp32'],
                    'scheme': ['sym',],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax']
                },
            }
        }
        self._query_quantizable_ops(model)
        capability['opwise'] = self.quantizable_op_details
        return capability

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        quantized_model = os.path.join(os.getcwd(), "tf_quantized.pb")

        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   quantized_model,
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   data_loader=dataloader)
        return converter.inspect_tensor(op_list, iteration_list)
