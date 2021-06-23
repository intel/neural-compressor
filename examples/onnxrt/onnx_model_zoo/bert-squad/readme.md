# Evaluate performance of ONNX Runtime(MobileBERT) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task. 

### Environment
onnx: 1.7.0
onnxruntime: 1.6.0+

### Prepare dataset
You should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

### Prepare model
Download pretrained bert model. We will refer to `vocab.txt` file.

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

Download BERT-Squad from [onnx model zoo](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad).
```bash
wget https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx
```

Update BERT-Squad model opset version to 12 due to lpot requirement.

```python
import onnx
import numpy as np
from onnx import shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import version_converter, helper, optimizer
from onnx import numpy_helper
import random

model = onnx.load('/path/to/bertsquad-10.onnx')    
onnx_domain = "ai.onnx"
onnx_op_set_version = 12

model.ir_version = 3
opset_info = next((opset for opset in model.opset_import if opset.domain == '' or opset.domain == onnx_domain), None)
if opset_info is not None:
    model.opset_import.remove(opset_info)
model.opset_import.extend([onnx.helper.make_opsetid("ai.onnx", onnx_op_set_version)])

onnx.save(model, '/path/to/bert_SQuAD.onnx')        
```


### Evaluating
To evaluate the model, run `main.py` with the path to the model:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=/path/to/SQuAD/dataset \
                   --config=bert.yaml
```


