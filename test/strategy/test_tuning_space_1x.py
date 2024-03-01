import unittest
from copy import deepcopy

from neural_compressor.conf.dotdict import DotDict
from neural_compressor.experimental.strategy.utils.tuning_space import TuningSpace
from neural_compressor.utils import logger

op_cap = {
    # op have both weight and activation and support static/dynamic/fp32
    ("op_name1", "op_type1"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "dynamic",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    # op have both weight and activation and support static/dynamic/fp32
    ("op_name2", "op_type1"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "dynamic",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    # op have both weight and activation and support static/fp32
    ("op_name3", "op_type2"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax", "kl"],
            },
        },
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    # op only have activation and support dynamic/fp32
    ("op_name4", "op_type3"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
        },
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "dynamic",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax"],
            },
        },
        {
            "activation": {"dtype": "fp32"},
        },
    ],
}


op_cap2 = {
    # The granularity of op activation do not support per_tensor.
    ("op_name4", "op_type1"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
    ]
}


class TestTuningSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.capability = {"calib": {"calib_sampling_size": [1, 10, 50]}, "op": deepcopy(op_cap)}
        # for optype1,'algorithm': ['minmax', 'kl'] -> ['minmax']
        self.optype_wise_user_config = {
            "op_type1": {
                "activation": {
                    "algorithm": ["minmax"],
                    "granularity": ["per_channel", "per_tensor"],
                }
            }
        }
        self.model_wise_user_config = {
            "activation": {
                "granularity": ["per_channel"],
            }
        }
        # fallback op_name4
        self.op_wise_user_config = {
            "op_name4": {
                "activation": {
                    "dtype": ["fp32"],
                }
            }
        }

        self.op_wise_user_config2 = {
            "op_name4": {
                "activation": {
                    "granularity": ["per_tensor"],
                }
            }
        }

        self.capability2 = {"calib": {"calib_sampling_size": [1, 10]}, "op": deepcopy(op_cap2)}

    def test_tuning_space_merge_op_wise_not_exist(self):
        # op-wise
        conf = {
            "usr_cfg": {
                "quantization": {
                    "op_wise": deepcopy(self.op_wise_user_config2),
                }
            }
        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability2), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())

    def test_tuning_space_creation(self):
        conf = None
        # Test the creation of tuning space
        tuning_space = TuningSpace(self.capability, conf)
        logger.debug(tuning_space.root_item.get_details())
        # ops supported static
        static_items = tuning_space.query_items_by_quant_mode("static")
        static_items_name = [item.name for item in static_items]
        self.assertEqual(set(static_items_name), set(op_cap.keys()))
        # ops supported dynamic
        dynamic_items = tuning_space.query_items_by_quant_mode("dynamic")
        dynamic_items_name = [item.name for item in dynamic_items]
        all_items_name = list(op_cap.keys())
        all_items_name.remove(("op_name3", "op_type2"))
        self.assertEqual(set(dynamic_items_name), set(all_items_name))
        # ops supported fp32
        fp32_items = tuning_space.query_items_by_quant_mode("fp32")
        fp32_items_name = [item.name for item in fp32_items]
        self.assertEqual(set(fp32_items_name), set(op_cap.keys()))
        # all optype
        self.assertEqual(list(tuning_space.op_type_wise_items.keys()), ["op_type1", "op_type2", "op_type3"])

    def test_tuning_space_merge_model_wise(self):
        # Test merge with user config, model-wise, optype-wise, op-wise
        # model-wise
        self.capability = {"calib": {"calib_sampling_size": [1, 10, 50]}, "op": op_cap}
        conf = {
            "usr_cfg": {
                "quantization": {
                    "model_wise": self.model_wise_user_config,
                }
            }
        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        found_per_tensor = False
        for quant_mode in ["static", "dynamic"]:
            for op_item in tuning_space2.query_items_by_quant_mode(quant_mode):
                for path in tuning_space2.ops_path_set[op_item.name]:
                    mode_item = tuning_space2.query_quant_mode_item_by_full_path(op_item.name, path)
                    act_algo_item = mode_item.get_option_by_name(("activation", "granularity"))
                    if act_algo_item and "per_tensor" in act_algo_item.options:
                        found_per_tensor = True
                        break
        self.assertFalse(found_per_tensor)

    def test_tuning_space_merge_optype_wise(self):
        # optype-wise
        conf = {
            "usr_cfg": {
                "quantization": {
                    "optype_wise": self.optype_wise_user_config,
                }
            }
        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        found_act_algo_kl_optype1 = False
        found_act_algo_kl_others = False
        for quant_mode in ["static", "dynamic"]:
            for op_item in tuning_space2.query_items_by_quant_mode(quant_mode):
                for path in tuning_space2.ops_path_set[op_item.name]:
                    mode_item = tuning_space2.query_quant_mode_item_by_full_path(op_item.name, path)
                    act_algo_item = mode_item.get_option_by_name(("activation", "algorithm"))
                    if act_algo_item and op_item.name[1] == "op_type1" and "kl" in act_algo_item.options:
                        found_act_algo_kl_optype1 = True
                        break
                    if act_algo_item and op_item.name[1] != "op_type1" and "kl" in act_algo_item.options:
                        found_act_algo_kl_others = True
        self.assertFalse(found_act_algo_kl_optype1)
        self.assertTrue(found_act_algo_kl_others)

    def test_tuning_space_merge_op_wise(self):
        # op-wise
        conf = {
            "usr_cfg": {
                "quantization": {
                    "op_wise": self.op_wise_user_config,
                }
            }
        }
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        found_quant_op_name4 = False
        found_fp32_op_name4 = False
        for quant_mode in ["static", "dynamic"]:
            for item in tuning_space2.query_items_by_quant_mode(quant_mode):
                if "op_name4" in item.name:
                    found_quant_op_name4 = True
                    break

        for item in tuning_space2.query_items_by_quant_mode("fp32"):
            if "op_name4" in item.name:
                found_fp32_op_name4 = True
                break
        self.assertFalse(found_quant_op_name4)
        self.assertTrue(found_fp32_op_name4)

    def test_tuning_space_merge_model_wise_and_opty_wise(self):
        # Test mode-wise + optype-wise
        conf = {
            "usr_cfg": {
                "quantization": {
                    "model_wise": self.model_wise_user_config,
                    "optype_wise": self.optype_wise_user_config,
                }
            }
        }
        # the optype_wise config will overwrite the model-wise config
        conf = DotDict(conf)
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        found_per_tensor = False
        for quant_mode in ["static", "dynamic"]:
            for op_item in tuning_space2.query_items_by_quant_mode(quant_mode):
                for path in tuning_space2.ops_path_set[op_item.name]:
                    mode_item = tuning_space2.query_quant_mode_item_by_full_path(op_item.name, path)
                    act_algo_item = mode_item.get_option_by_name(("activation", "granularity"))
                    if act_algo_item and "per_tensor" in act_algo_item.options:
                        found_per_tensor = True
                        break
        self.assertTrue(found_per_tensor)


if __name__ == "__main__":
    unittest.main()
