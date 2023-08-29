import unittest
from collections import OrderedDict
from copy import deepcopy

from neural_compressor.strategy.utils.tuning_sampler import (
    BlockFallbackTuningSampler,
    FallbackTuningSampler,
    ModelWiseTuningSampler,
    OpTypeWiseTuningSampler,
    OpWiseTuningSampler,
)
from neural_compressor.strategy.utils.tuning_space import TuningSpace
from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig

op_cap = {
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
    ("op_name3", "op_type2"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel"],
            },
            "weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_channel"]},
        },
        {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}},
    ],
    ("op_name4", "op_type3"): [
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "static",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
            },
        },
        {
            "activation": {
                "dtype": ["int8"],
                "quant_mode": "dynamic",
                "scheme": ["sym"],
                "granularity": ["per_channel", "per_tensor"],
            },
        },
        {
            "activation": {"dtype": "fp32"},
        },
    ],
}


block_wise = [
    [("op_name2", "op_type1"), ("op_name4", "op_type3")],
    [("op_name4", "op_type3"), ("op_name1", "op_type1")],
]


class TestTuningSampler(unittest.TestCase):
    def test_tuning_sampler(self):
        capability = {"calib": {"calib_sampling_size": [1, 10, 50]}, "op": deepcopy(op_cap)}
        conf = None
        tuning_space = TuningSpace(capability, conf)

        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == "op":
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, "fp32", tuning_space)
                print(initial_op_tuning_cfg[item.name])
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
        self.assertEqual(len(list(op_wise_tuning_sampler)), 128)
        optype_wise_tuning_sampler = OpTypeWiseTuningSampler(
            deepcopy(tuning_space), [], [], op_item_dtype_dict, initial_op_tuning_cfg
        )
        cfg_lst = list(optype_wise_tuning_sampler)
        self.assertEqual(len(cfg_lst), 16)
        model_wise_tuning_sampler = ModelWiseTuningSampler(
            deepcopy(tuning_space), [], [], op_item_dtype_dict, initial_op_tuning_cfg
        )
        model_wise_pool = []
        best_tune_cfg = None
        for tune_cfg in model_wise_tuning_sampler:
            best_tune_cfg = tune_cfg
            model_wise_pool.append(tune_cfg)
        self.assertEqual(len(model_wise_pool), 8)

        # fallback test
        quant_ops = quant_mode_wise_items.get("static", [])
        quant_ops += quant_mode_wise_items.get("dynamic", [])
        target_dtype = "fp32"
        target_type_lst = tuning_space.query_items_by_quant_mode(target_dtype)
        fallback_items_lst = [item for item in quant_ops if item in target_type_lst]
        if fallback_items_lst:
            print(f"Start to fallback op to {target_dtype} one by one.")
        fallback_items_name_lst = [item.name for item in fallback_items_lst]
        op_dtypes = OrderedDict(zip(fallback_items_name_lst[::-1], [target_dtype] * len(fallback_items_name_lst)))
        initial_op_tuning_cfg = deepcopy(best_tune_cfg)
        fallback_sampler = FallbackTuningSampler(
            tuning_space,
            tuning_order_lst=[],
            initial_op_tuning_cfg=initial_op_tuning_cfg,
            op_dtypes=op_dtypes,
            accumulate=False,
        )
        fallback_cnt = []
        fp32_lst = []
        for op_cfgs in fallback_sampler:
            cnt = 0
            for op_name, op_cfg in op_cfgs.items():
                op_state = op_cfg.get_state()
                if "fp32" == op_state["activation"]["dtype"] and (
                    "fp32" == op_state["weight"]["dtype"] if "weight" in op_state else True
                ):
                    cnt = cnt + 1
                    fp32_lst.append(op_name)
            fallback_cnt.append(cnt)
        self.assertListEqual(fallback_cnt, [1, 1, 1, 1])
        self.assertListEqual(fp32_lst, fallback_items_name_lst[::-1])

        fallback_sampler_acc = FallbackTuningSampler(
            tuning_space,
            tuning_order_lst=[],
            initial_op_tuning_cfg=initial_op_tuning_cfg,
            op_dtypes=op_dtypes,
            accumulate=True,
        )
        fallback_cnt = []
        for op_cfgs in fallback_sampler_acc:
            cnt = 0
            for op_name, op_cfg in op_cfgs.items():
                op_state = op_cfg.get_state()
                if "fp32" == op_state["activation"]["dtype"] and (
                    "fp32" == op_state["weight"]["dtype"] if "weight" in op_state else True
                ):
                    cnt = cnt + 1
            fallback_cnt.append(cnt)
        self.assertListEqual(fallback_cnt, [2, 3, 4])

    def test_block_sampler(self):
        capability = {"calib": {"calib_sampling_size": [1, 10, 50]}, "op": deepcopy(op_cap), "block_wise": block_wise}
        conf = None
        tuning_space = TuningSpace(capability, conf)
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == "op":
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, "fp32", tuning_space)
                print(initial_op_tuning_cfg[item.name])
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

        op_block_lst = capability.get("block_wise", [])
        if op_block_lst:
            # Fallback block by block
            target_type_lst = set(tuning_space.query_items_by_quant_mode("fp32"))
            fallback_items_lst = [item for item in target_type_lst]
            op_block_fallback_lst = []
            for op_block_index, op_block in enumerate(op_block_lst):
                op_block_fallback_lst.append(op_block)
            block_fallback_sampler = BlockFallbackTuningSampler(
                tuning_space=tuning_space,
                tuning_order_lst=[],
                initial_op_tuning_cfg=initial_op_tuning_cfg,
                op_block_lst=op_block_fallback_lst,
                accumulate=False,
                target_dtype="fp32",
            )
            self.assertEqual(2, len(list(block_fallback_sampler)))


if __name__ == "__main__":
    unittest.main()
