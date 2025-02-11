Note for static quantize DeepSeek model

## Prerequisite
```
pip install -r requirements.txt
```

## Usage

- Option 1. (Rec)
```bash
python quant.py --model_path /path/to/DeepSeek/R1/BF16/ --qmodel_path /path/to/DeepSeek/R1/Dynamic-FP8
```

- Option 2. handle weights only (If the DRAM size is less than ~700 GB)
```bash
python quant.py --model_path /path/to/DeepSeek/R1/BF16/ --qmodel_path /path/to/DeepSeek/R1-Dynamic-FP8 --low_cpu_mem
```

> [!NOTE]
> - Skip quantize `lm-head`.
> - `WEIGHT_BACKOFF = 0.5`
> - `SCALE_DTYPE = torch.bfloat16`


## Example
1. Name convention:
    - weight scale name: `prefix.scale_weight`
    - input scale name: `prefix.scale_input` (for static only)
2. A json file mapping from tensor name to safetensor file name.

```python
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5, bias=False)

    def forward(self, inp):
        x1 = self.fc1(inp)
        return x1
```

```bash
1. state dict
{
    "fc1.weight": torch.Tensor(...),
    "fc1.scale_weight": torch.Tensor(...),
    "fc1.scale_input": torch.Tensor(...),
}

2. json file, `model.safetensors.index.json`
{
    "fc1.weight": "qmodel.safetensors",
    "fc1.scale_weight": "qmodel.safetensors",
    "fc1.scale_input": "qmodel.safetensors"
}
```

