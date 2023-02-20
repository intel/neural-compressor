from neural_compressor.strategy.utils.tuning_space import TuningItem, TuningSpace
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.utils import logger
from copy import deepcopy
import unittest

op_cap = {
    # op have both weight and activation and support static/dynamic/fp32
    ('op_name1', 'op_type1'): [
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': ['int4'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['uint4'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'dynamic',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': 'bf16'
                },
            'weight':
                {
                    'dtype': 'bf16'
                }
        },
        {
            'activation':
                {
                    'dtype': 'fp32'
                },
            'weight':
                {
                    'dtype': 'fp32'
                }
        },
    ],
    # op have both weight and activation and support static/dynamic/fp32
    ('op_name2', 'op_type1'): [
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'dynamic',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': 'fp32'
                },
            'weight':
                {
                    'dtype': 'fp32'
                }
        },
    ],
    # op have both weight and activation and support static/fp32
    ('op_name3', 'op_type2'): [
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel'],
                    'algorithm': ['minmax', 'kl']
                }
        },
        {
            'activation':
                {
                    'dtype': 'fp32'
                },
            'weight':
                {
                    'dtype': 'fp32'
                }
        },
    ],
    # op have both weight and activation and support dynamic/fp32
    ('op_name4', 'op_type3'): [
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
        },
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'dynamic',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax']
                },
        },
        {
            'activation':
                {
                    'dtype': 'fp32'
                },
        },
    ]
}


op_cap2 = {
    # The granularity of op activation do not support per_tensor.
    ('op_name4', 'op_type1'): [
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },]
}


class TestTuningSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.capability = {
            'calib': {'calib_sampling_size': [1, 10, 50]},
            'op': deepcopy(op_cap)
        }

        self.optype_wise_user_config = {
            'op_type1': {
                'activation': {
                    'algorithm': ['minmax'],
                    'granularity': ['per_channel', 'per_tensor'],
                }
            }
        }
        self.model_wise_user_config = {
            'activation': {
                'granularity': ['per_channel'],
            }
        }

        self.op_wise_user_config = {
            'op_name1': {
                'activation': {
                    'dtype': ['bf16']
                },
                'weight': {
                    'dtype': ['bf16']
                }
            },
            'op_name2': {
                'activation': {
                    'dtype': ['fp32']
                },
                'weight': {
                    'dtype': ['fp32']
                }
            }, # Fallback op_name2 to fp32 by manually, only keep fp32 capability
            'op_name3': {
                'activation': {
                    'dtype': ['int8']
                },
                'weight': {
                    'dtype': ['int8']
                }
            }, # Only keep int8 capability
            'op_name4': {
                'activation': {
                    'granularity': ['per_channel'],
                }
            }
        }

        self.op_wise_user_config2 = {
            'op_name4': {
                'activation': {
                    'granularity': ['per_tensor'],
                }
            }
        }
        
        self.capability2 = {
            'calib': {'calib_sampling_size': [1, 10]},
            'op': deepcopy(op_cap2)
        }

    

    def test_tuning_parse_helper(self):
        # op-wise
        conf = {
            'usr_cfg': {
                'quantization':{}
            }

        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        
    def test_sampler(self):
        from neural_compressor.strategy.utils.tuning_sampler import OpTypeWiseTuningSampler
        from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig
        tuning_space2 = TuningSpace(deepcopy(self.capability), {})
        logger.debug(tuning_space2.root_item.get_details())
        from neural_compressor.strategy.utils.tuning_space import get_op_mode_by_query_order
        from neural_compressor.strategy.utils.constant import static_query_order as query_order
        op_item_dtype_dict = get_op_mode_by_query_order(tuning_space2, query_order)
        print(op_item_dtype_dict)
        # Initialize the tuning cfg 
        initial_op_tuning_cfg = {}
        for item in tuning_space2.root_item.options:
            if item.item_type == 'op':
                op_name, op_type = item.name
                opconfig = OpTuningConfig(op_name, op_type, 'fp32', tuning_space2)
                initial_op_tuning_cfg[item.name] = opconfig
                logger.info(opconfig)

        optype_wise_tuning_sampler = OpTypeWiseTuningSampler(deepcopy(tuning_space2), [], [], 
                                                             op_item_dtype_dict, initial_op_tuning_cfg)

    def test_tuning_space_merge_op_wise(self):
        # op-wise
        conf = {
            'usr_cfg': {
                'quantization': {
                    'op_wise': self.op_wise_user_config,
                }
            }

        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        # found_quant_op_name4 = False
        # found_fp32_op_name4 = False
        # for quant_mode in ['static', 'dynamic']:
        #     for item in tuning_space2.query_items_by_quant_mode(quant_mode):
        #         if 'op_name4' in item.name:
        #             found_quant_op_name4 = True
        #             break

        # for item in tuning_space2.query_items_by_quant_mode('fp32'):
        #     if 'op_name4' in item.name:
        #         found_fp32_op_name4 = True
        #         break
        # self.assertFalse(found_quant_op_name4)
        # self.assertTrue(found_fp32_op_name4)


if __name__ == "__main__":
    unittest.main()
