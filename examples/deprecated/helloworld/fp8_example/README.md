### Usage demo:

#### two steps to get quantized model

```diff
import torch
+ from neural_compressor.torch.quantization import FP8Config, convert, prepare, finalize_calibration
import habana_frameworks.torch.core as htcore

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2

model = M().eval()

+ config = FP8Config.from_json_file(args.quant_config) # args.quant_config is the path of json file

+ if config.measure:
+   model = prepare(model, config)

+ if config.quantize:
+     htcore.hpu_initialize()
+     model = convert(model, config)

# user code run
with torch.no_grad():
    model.to("hpu")
    output = model(torch.randn(1, 10).to("hpu"))
    print(output)

+ if config.measure:
+    finalize_calibration(model)
```


Whole script and config refer to [sample_two_steps.py](sample_two_steps.py), [maxabs_measure.json](maxabs_measure.json) and [maxabs_quant.json](maxabs_quant.json).

First, measure the tensor quantization statistic:
```shell
python sample_two_steps.py --quant_config=maxabs_measure.json
```

Then quantize the model based on previous measurements:
```shell
python sample_two_steps.py --quant_config=maxabs_quant.json
```

#### one step to get quantized model

```diff
import torch
+ from neural_compressor.torch.quantization import FP8Config, convert, prepare, finalize_calibration
import habana_frameworks.torch.core as htcore

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

+ config = FP8Config.from_json_file(args.quant_config) # args.quant_config is the path of json file
+ model = prepare(model, config)

# user code run to do calibration
with torch.no_grad():
    output = model(torch.randn(1, 10).to("hpu"))
    print(output)

+ finalize_calibration(model)
+ model = convert(model)

# user code to run benchmark for quantized model
with torch.no_grad():
    output = model(torch.randn(1, 10).to("hpu"))
    print(output)
```

Whole script and config refer to [sample_one_step.py](sample_one_step.py).

```shell
python sample_one_step.py --quant_config=quant_config.json
```
