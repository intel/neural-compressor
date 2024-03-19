file tree:
```bash
├── neural_compressor
│   ├── common
│   ├── torch
│   │   ├── algorithms
│   │   ├── quantization
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── quantize.py
│   │   │   └── quantizer
│   │   │       ├── base_quantizer.py
│   │   │       ├── guadi_quantizer.py
│   │   │       └── __init__.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── __init__.py
│   └── version.py
├── test
│   └── sample.py

```

usage demo:

```diff
import torch
+ from neural_compressor.torch import FP8QuantConfig, prepare, convert, save_calib

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2

model = M().to("hpu")

+ quant_config = FP8QuantConfig()
+ model = prepare(model, quant_config) # prepare the model for quantization if needed

# reuse user's eval func to do calibration
eval_func(model)

+ save_calib(model, quant_config) # save calibration results to local file if needed
+ model = convert(model, quant_config) # convert the model to a quantized model
```
