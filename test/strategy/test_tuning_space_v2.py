import unittest
from copy import deepcopy

from neural_compressor.conf.dotdict import DotDict
from neural_compressor.strategy.utils.tuning_space import TuningItem, TuningSpace
from neural_compressor.utils import logger

op_cap = {
    # op1 have both weight and activation and support static/dynamic/fp32/b16
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
                "dtype": ["int4"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["uint4"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
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
        {"activation": {"dtype": "bf16"}, "weight": {"dtype": "bf16"}},
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    # op2 have both weight and activation and support static/dynamic/fp32
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
    # op3 have both weight and activation and support int4
    ("op_name3", "op_type3"): [
        {
            "activation": {
                "dtype": ["int4"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
                "algorithm": ["minmax", "kl"],
            },
            "weight": {"dtype": ["int4"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
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
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    # op4 have tuple name as IPEX
    (("op_name4", 0), "op_type4"): [
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
    # op5, weight only
    ("op_name5", "op_type5"): [
        {
            "activation": {
                "dtype": ["fp32"],
                "quant_mode": "static",
            },
            "weight": {"dtype": ["int4"], "scheme": ["sym"], "granularity": ["per_channel", "per_tensor"]},
        },
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
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
}


class TestTuningSpaceV2(unittest.TestCase):
    def setUp(self) -> None:
        self.capability = {"calib": {"calib_sampling_size": [1, 10, 50]}, "op": deepcopy(op_cap)}

        self.op_wise_user_cfg_for_fallback = {
            "op_name1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
            ("op_name4", 0): {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
        }

    def test_tuning_sampler_int4(self):
        # op-wise
        conf = {}
        conf = DotDict(conf)
        # test space construction
        tuning_space = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space.root_item.get_details())
        found_int4_activation = False
        found_int4_weight = False
        op3_act_item = tuning_space.query_quant_mode_item_by_full_path(
            ("op_name3", "op_type3"), ("static", "activation")
        )
        for dtype_item in op3_act_item.options:
            if dtype_item.name == "int4":
                found_int4_activation = True
        self.assertTrue(found_int4_activation)
        op3_weight_item = tuning_space.query_quant_mode_item_by_full_path(
            ("op_name3", "op_type3"), ("static", "weight")
        )
        for dtype_item in op3_weight_item.options:
            if dtype_item.name == "int4":
                found_int4_weight = True
        self.assertTrue(found_int4_weight)

    def test_sampler_int4(self):
        # test sampler
        from collections import OrderedDict

        from neural_compressor.strategy.utils.tuning_sampler import LowerBitsSampler, OpWiseTuningSampler
        from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig

        # op-wise
        conf = {}
        conf = DotDict(conf)
        # test space construction
        tuning_space = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        # test tuning item basic functionality
        logger.debug(str(tuning_space.root_item))
        logger.debug(tuning_space.root_item.get_options_name())
        tuning_space.root_item.remove("fake_options")
        logger.debug(tuning_space.root_item.get_details())
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == "op":
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, "fp32", tuning_space)
        quant_mode_wise_items = OrderedDict()
        from neural_compressor.strategy.utils.constant import auto_query_order as query_order

        pre_items = set()
        for quant_mode in query_order:
            items = tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = [item for item in items if item not in pre_items]
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

        op_wise_tuning_sampler = OpWiseTuningSampler(
            deepcopy(tuning_space), [], [], op_item_dtype_dict, initial_op_tuning_cfg
        )
        op3 = ("op_name3", "op_type3")
        for tune_cfg in op_wise_tuning_sampler:
            op_cfg = tune_cfg[op3].get_state()
            act_dtype = op_cfg["activation"]["dtype"]
            weight_dtype = op_cfg["weight"]["dtype"]
            self.assertTrue(act_dtype == weight_dtype == "int4")

        int4_ops = tuning_space.collect_op_by_quant_bits("int4")
        for op in int4_ops:
            op_item_dtype_dict[op.name] = "int4"
        lower_bits_sampler = LowerBitsSampler(
            deepcopy(tuning_space), [], initial_op_tuning_cfg, op_item_dtype_dict, accumulate=False, skip_first=True
        )
        op3 = ("op_name5", "op_type5")
        for tune_cfg in lower_bits_sampler:
            op_cfg = tune_cfg[op3].get_state()
            act_dtype = op_cfg["activation"]["dtype"]
            weight_dtype = op_cfg["weight"]["dtype"]
            logger.debug(op_cfg)
            self.assertTrue((weight_dtype == "int4" and act_dtype == "fp32") or (act_dtype == weight_dtype == "fp32"))

    def test_tuning_space_merge_op_wise(self):
        # op-wise
        conf = {
            "op_name_dict": self.op_wise_user_cfg_for_fallback,
        }
        conf = DotDict(conf)
        # test fallback
        tuning_space2 = TuningSpace(deepcopy(self.capability), deepcopy(conf))
        logger.debug(tuning_space2.root_item.get_details())
        op_name1_only_fp32 = True
        op_name4_only_fp32 = True
        for quant_mode in ["static", "dynamic"]:
            for item in tuning_space2.query_items_by_quant_mode(quant_mode):
                if item.name[0] == "op_name1":
                    op_name1_only_fp32 = False
                if item.name[0] == ("op_name4", 0):
                    op_name4_only_fp32 = False
        self.assertTrue(op_name1_only_fp32)
        self.assertTrue(op_name4_only_fp32)


if __name__ == "__main__":
    unittest.main()
