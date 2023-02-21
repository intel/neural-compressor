from neural_compressor.strategy.utils.tuning_space import TuningItem, TuningSpace
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.utils import logger
from copy import deepcopy
import unittest

op_cap = {
    # op1 have both weight and activation and support static/dynamic/fp32/b16
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
    # op2 have both weight and activation and support static/dynamic/fp32
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
}

class TestTuningSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.capability = {
            'calib': {'calib_sampling_size': [1, 10, 50]},
            'op': deepcopy(op_cap)
        }
        
        self.op_wise_user_cfg_for_fallback = {
            'op_name1': {
                'activation': {
                    'dtype': ['fp32']
                },
                'weight': {
                    'dtype': ['fp32']
                }
            },
        }

    def test_tuning_space_merge_op_wise(self):
        # op-wise
        conf = {
            'usr_cfg': {
                'quantization': {
                    'op_wise': self.op_wise_user_cfg_for_fallback,
                }
            }

        }
        conf = DotDict(conf)
        # test fallback
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        op_name1_only_fp32 = True
        for quant_mode in ['static', 'dynamic']:
            for item in tuning_space2.query_items_by_quant_mode(quant_mode):
                if item.name[0] == 'op_name1':
                    op_name1_only_fp32 = False
        self.assertTrue(op_name1_only_fp32)
        # test options merge
        


if __name__ == "__main__":
    unittest.main()
