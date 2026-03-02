import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import ScaleMethodString

from ..tester import TestVector, run_accuracy_test


class LinearBlock(torch.nn.Module):
    def __init__(self):
        super(LinearBlock, self).__init__()
        self.linear_ = torch.nn.Linear(2, 2, bias=True)
        self.linear_.weight = torch.nn.Parameter(torch.arange(0.0, 4.0).reshape(2, 2))
        self.linear_.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.linear_(x)


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super(TinyBlock, self).__init__()
        self.pre_linear = torch.nn.Linear(2, 2, bias=False)
        self.pre_linear.weight = torch.nn.Parameter(torch.ones((2, 2)) / 4)

        self.linear1 = LinearBlock()
        self.post_linear = torch.nn.Linear(2, 2, bias=False)
        self.post_linear.weight = torch.nn.Parameter(torch.ones((2, 2)) / 4)
        self.linear2 = LinearBlock()

    def forward(self, x):
        x = self.pre_linear(x)
        x = self.linear1(x)
        x = self.post_linear(x)
        x = self.linear2(x)
        x = x.sum()
        return x


class TinyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        block = TinyBlock()

        # no kernel inject - currently only works on Habana's DeepSpeed fork!
        # these layers will be switched to LinearAllReduce.
        injection_policy = {TinyBlock: ("linear1.linear_", "linear2.linear_")}

        # Initialize deepspeed on model creation
        import deepspeed

        block = deepspeed.init_inference(
            block,
            injection_policy=injection_policy,
            **kwargs,
        )
        self.block = block.module

    def forward(self, x):
        return self.block(x)


def get_test_vectors(dtype: torch.dtype) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(1, 2).to(device="hpu", dtype=dtype)],
    )


# @pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32])
# TODO: float32 doesn't work - WHY?
# TODO: add ticket
@pytest.mark.deepspeed
@pytest.mark.parametrize("hp_dtype", [torch.bfloat16])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_deepspeed_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype):
    world_size = 1
    run_accuracy_test(
        module_class=TinyModel,
        test_vectors=get_test_vectors(dtype=hp_dtype),
        lp_dtype=lp_dtype,
        scale_method=ScaleMethodString.MAXABS_HW,
        module_kwargs={"dtype": hp_dtype, "mp_size": world_size},
    )
