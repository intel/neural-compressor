### File tree:
```bash
├── neural_compressor
│   ├── common
│   ├── torch
│   │   ├── algorithms
│   │   │   └── habana_fp8
│   │   │         ├──__init__.py
│   │   │         ├── fp8_quant.py # FP8Quantizer
│   │   │         ├── helper_modules.py # revert patched module to origin module
│   │   │         └── common.py
|   |   ├── base_algo.py # base class for different algos
│   │   └── __init__.md
│   │   ├── quantization
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── quantize.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── __init__.py
│   └── version.py
├── test
│   ├── sample.py
│   ├── sample_one_step.py
│   ├── maxabs_measure.json
│   ├── maxabs_quant.py
│   └── quant_config.py


```

### Usage demo:

#### two steps to get quantized model (HQT flow)

```diff
import torch
+ from neural_compressor.torch import FP8QuantConfig, convert, prepare, finalize_calibration
import habana_frameworks.torch.core as htcore
htcore.hpu_set_env()

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

+ config = FP8QuantConfig.from_json_file(args.quant_config) # args.quant_config is the path of json file

+ if config.calibrate:
+   model = prepare(model, config)

+ if config.quantize:
+     model = convert(model, config)

# user code run
input = torch.randn(1, 10)
model(input.to("hpu"))

+ if config.calibrate:
+    finalize_calibration(model)
```


Whole script and config refer to [sample.py](../../test/sample.py), [maxabs_measure.json](../../test/maxabs_measure.json) and [maxabs_quant.json](../../test/maxabs_quant.json).

First, measure the tensor quantization statistic:
```shell
python sample.py --quant_config=maxabs_measure.json
```

Then quantize the model based on previous measurements:
```shell
python sample.py --quant_config=maxabs_quant.json
```

#### one step to get quantized model (INC flow)

```diff
import torch
+ from neural_compressor.torch import FP8QuantConfig, convert, prepare, finalize_calibration
import habana_frameworks.torch.core as htcore
htcore.hpu_set_env()

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

+ config = FP8QuantConfig.from_json_file(args.quant_config) # args.quant_config is the path of json file
+ model = prepare(model, config)

# user code run
input = torch.randn(1, 10)
model(input.to("hpu"))

+ finalize_calibration(model)
+ model = convert(model)
```

Whole script and config refer to [sample_one_step.py](../../test/sample_one_step.py).

```shell
python sample_one_step.py --calib_result ./hqt_output/measure --quant_config=quant_config.json
```
