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
│   │   │   └── backend
│   │   │       ├── base_backend.py
│   │   │       ├── hqt_backend.py
│   │   │       └── __init__.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── __init__.py
│   └── version.py
├── test
│   └── sample.py

```

usage demo:

1. One step to get a quantized model

```diff
import torch
+ from neural_compressor.torch import prepare, convert, save_calibration_result

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

+ model = prepare(model, quant_config="quant_config.json") # prepare the model for quantization

# reuse user's eval func to do calibration
eval_func(model)

+ save_calibration_result(model) # save calibration results to local file
+ model = convert(model) # convert the model to a quantized model

# reuse user's eval func to do evaluation
eval_func(model)
```

2. Two step to get a quantized model

You need to run this script twice to perform the calibration and quantization steps separately.

```diff
import torch
+ from neural_compressor.torch import prepare, convert, save_calibration_result

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

+ model = prepare(model, quant_config="quant_config.json") # prepare the model for quantization if needed
+ model = convert(model) # convert the model to a quantized model if needed

# reuse user's eval func to do calibration/evaluation
eval_func(model)

+ save_calibration_result(model) # save calibration results to local file if needed

```
