# Evaluate performance of ONNX Runtime(ResNet 50) 

This example load an image classification model exported from PyTorch and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.7.0
onnxruntime: 1.5.2

### Prepare model
Please refer to [pytorch official guide](https://pytorch.org/docs/stable/onnx.html) for detailed model export. The following is a simple example:

```python
import torch
import torchvision
model = torchvision.models.resnet50()
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",           # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to, please ensure at least 11.
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
```

### Evaluating
To evaluate the model, run `main.py` with the path to the model:

```cmd
python main.py --model_path path/to/model  # model pat as *.onnx
               --benchmark                 # (Optional) whether to get benchmark results
               --tune                      # (Optional) whether to tune a model meeting requirements
               --config resnet50_v1_5.yaml # (Needed if tune or benchmark)
```
### Advanced 
Usually we need to bind the program to specific cores like 4 cores to get performance under real production environments.   
**for linux**
```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=4
numactl --physcpubind=0-3 --membind=0 python main.py --model_path path/to/model --benchmark
--tune  --config resnet50_v1_5.yaml 
```

**for windows**
```cmd
start /wait  /b /node /affinity f python main.py --model_path path/to/model --benchmark
--tune  --config resnet50_v1_5.yaml 
```
You can refer to [windows doc](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/start) for detailed instruction.

