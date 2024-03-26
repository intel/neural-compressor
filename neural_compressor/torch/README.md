### File tree:
```bash
├── neural_compressor
│   ├── common
│   ├── torch
│   │   ├── algorithms
│   │   │   └── habana_fp8
│   │   │         ├──__init__.py
│   │   │         └── common.py
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
│   ├── calib.json
│   └── quantize.py


```

### Usage demo:

#### python code:

```diff
import torch
+ from neural_compressor.torch import prepare, convert, save

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

+ if args.calib:
+    model = prepare(model, args.quant_config)

+ if args.quantize:
+    model = convert(model, args.quant_config, args.calib_result)

eval_func(model)

+ save(model, args.calib_result)
```

#### config json:
```json
{
    "method": "HOOKS",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "whitelist": {"types": [], "names":  []},
    "blacklist": {"types": [], "names":  []},
    "dump_stats_path": "./hqt_output/measure"
}
```
> Note: `dump_stats_path` set in the json file will not work.

Whole scrip and config refer to [sample.py](../../test/sample.py) and [quant_config.json](../../test/quant_config.json).

First, measure the tensor quantization statistic:
```shell
python sample.py --calib --calib_result ./hqt_output/measure --quant_config=quant_config.json
```

Then quantize the model based on previous measurements:
```shell
python sample.py --quantize --calib_result ./hqt_output/measure --quant_config=quant_config.json
```
