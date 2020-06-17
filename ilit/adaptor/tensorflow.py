from .adaptor import adaptor_registry, Adaptor
from ..utils import LazyImport

from collections import OrderedDict
import os

tensorflow = LazyImport('tensorflow')

@adaptor_registry
class TensorFlowAdaptor(Adaptor):
    unify_op_type_mapping = {
            "Conv2D": "conv2d",
            "DepthwiseConv2dNative": "conv2d",
            "MaxPool": "pooling",
            "AvgPool": "pooling"
        }

    def __init__(self, input_output_info):
        super(TensorFlowAdaptor, self).__init__(input_output_info)

        self.op_wise_config = {}
        self.input_output_info = input_output_info

    def calibrate(self, model):
        return self.calibrate()

    def test(self, model):
        return self.test()

    def metric(self, model):
        return self.metric()


    def tuning_cfg_to_fw(self, tuning_cfg):
        # TODO add the op-wise config parse
        self.excluded_nodes = []
        for each_op_info in tuning_cfg['op']:
            op_name = each_op_info[0]

            if tuning_cfg['op'][each_op_info]['activation']['data_type'] == 'fp32':
                self.excluded_nodes.append(op_name)
                continue
            is_perchannel = tuning_cfg['op'][each_op_info]['activation']['granularity'] == 'perchannel'
            algo = tuning_cfg['op'][each_op_info]['activation']['algo']
            self.op_wise_config[op_name] = (is_perchannel, algo)


    def quantize(self, tune_cfg, model, data_loader):
        quantized_model = os.path.join(os.getcwd() + "tf_quantized.model")
        self.tuning_cfg_to_fw(tune_cfg)
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model, quantized_model, inputs=self.input_output_info['inputs'],
                                   outputs=self.input_output_info['outputs'], op_wise_config=self.op_wise_config, data_loader=data_loader)
        return converter.convert()

    def query_quantizable_ops(self, graph):
        '''
            Return: Op name/Op type mapping which saved in OrderDict.
        '''
        graph_def = graph.as_graph_def()
        tf_quantizable_op_type = ("Conv2D", "DepthwiseConv2dNative", "MaxPool", "AvgPool")
        conv_config = {
            'activation': {
                'data_type': ['uint8', 'fp32'],
                'algo': ['minmax', 'kl'],
                'mode': ['sym']
            },
            'weight': {
                'data_type': ['int8', 'fp32'],
                'algo': ['kl']
            }
        }
        non_conv_config = {
            'activation': {
                'data_type': ['uint8', 'fp32'],
                'algo': ['minmax', 'kl'],
                'mode': ['sym']
            },
        }

        self.quantizable_op_details = OrderedDict()
        for node in graph_def.node:
            if node.op in tf_quantizable_op_type:
                # each_op_info = {}
                # each_op_info['op_type'] = self.unify_op_type_mapping[node.op]
                # each_op_info['op_name'] = node.name
                # each_op_info['data_type'] = ['fp32', 'int8']
                # self.quantizable_op_details.append(each_op_info)
                self.quantizable_op_details[(
                    node.name, self.unify_op_type_mapping[node.op]
                )] = conv_config if self.unify_op_type_mapping[node.op].find(
                    "Conv") != -1 else non_conv_config
                self.op_wise_config[node.name] = (False, "minmax")

        return self.quantizable_op_details

    def query_fw_capability(self, model):
        capability = {
            'modelwise': {
                'activation': {
                    'data_type': ['uint8', 'fp32'],
                    'mode': ['sym'],
                    'granularity': ['per_tensor'],
                    'algo': ['minmax', 'kl']
                },
                'weight': {
                    'data_type': ['int8',  'fp32'],
                    'mode': ['sym',],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algo': ['minmax']
                },
            }
        }
        self.query_quantizable_ops(model)
        capability['opwise'] = self.quantizable_op_details
        return capability
        # capability = \
        #     {
        #     'algo':
        #         {
        #         'weight': ['minmax'],
        #         'activation': ['minmax', 'kl']
        #         },
        #     'activation':
        #         {
        #         'granularity': ['per_channel', ],
        #         # 'mode': ['sym', 'asym'],
        #         'mode': ['sym' ],
        #         'data_type':['int8']
        #         },
        #     'weight':
        #         {
        #         'granularity': ['per_channel','per_tensor'],
        #         # 'mode': ['sym', 'asym'],
        #         'mode': ['sym'],
        #         'data_type':['int8']
        #         }
        #     }
        # return capability

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        quantized_model = os.path.join(os.getcwd() + "tf_quantized.model")

        input_node_name = ['input']
        output_node_name = ['predict']
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   quantized_model,
                                   inputs=input_node_name,
                                   outputs=output_node_name,
                                   data_loader=dataloader)
        return converter.inspect_tensor(op_list, iteration_list)
