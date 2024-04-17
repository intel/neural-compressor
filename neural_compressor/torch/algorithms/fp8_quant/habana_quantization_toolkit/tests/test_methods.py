
import torch

import habana_quantization_toolkit
import habana_frameworks.torch.core as htcore


class TinyBlock(torch.nn.Module):

    def __init__(self):
        super(TinyBlock, self).__init__()
        self.pre_linear = torch.nn.Linear(3, 2, bias=False)
        self.pre_linear.weight = torch.nn.Parameter(torch.tensor([[0., 10., 20., ], [10., 20., 100.]]))


    def forward(self, x):
        x = self.pre_linear(x)
        x = x.sum()
        return x


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        self.block = TinyBlock()

    def forward(self, x):
        x = self.block(x)
        return x


model = TinyModel()
model.eval()
model = model.to('hpu').to(torch.bfloat16)
habana_quantization_toolkit.prep_model(model) # fp8 additions

with torch.no_grad():
    out_ones = model(torch.ones([2,3], dtype=torch.bfloat16).to('hpu'))
    print(out_ones)

    out_arange = model(torch.arange(6, dtype=torch.bfloat16).reshape([2,3]).to('hpu') * 10)
    print(out_arange)

    # fp8 additions
    habana_quantization_toolkit.finish_measurements(model)

