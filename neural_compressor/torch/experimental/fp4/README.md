# NV FP4 Estimator

## Introduction

Currently, we support [NVFP4_MODULE_MAPPING](./nvfp4.py) and modules in this mapping will be replaced with the `NVFP4` one. Both weights and activations will be quantized to `NVFP4`.

## Requirement

```shell
# Install auto-round from source
cd ~
git clone https://github.com/intel/auto-round.git
cd auto-round
pip install -e .
# Install neural-compressor from source
cd ~
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
git checkout xinhe/nvfp4
INC_PT_ONLY=1 pip install -e .
```

## Usage
```python
from neural_compressor.torch.experimental.fp4.nvfp4 import qdq_model
qdq_m = qdq_model(m)
```

## Example
```python
import torch


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

m = M()

with torch.no_grad():
    label = m(torch.randn(2, 10))
    print(label)

    from neural_compressor.torch.experimental.fp4.nvfp4 import qdq_model
    qdq_m = qdq_model(m)
    print(qdq_m)
    # M(
    #   (linear): NVFP4Linear(in_features=10, out_features=1, bias=True)
    # )

    out = m(torch.randn(2, 10))
    print(out)

```
