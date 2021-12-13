## Deploy and Integration

In this tutorial, we will deploy a TF/ONNX model using Engine inference OR through Manual customized yaml and weight binary to use Engine inference.

### 1. Deploy a TF/ONNX model using Engine inference

### Generate the Engine Graph through TF/ONNX model

Only support TensorFlow and ONNX model for now.

```
from engine.compile import compile
model = compile('/path/to/your/model')
model.save('/ir/path')   # Engine graph could be saved to path
```

Engine graph could be saved as yaml and weight bin.

### Run the inference by Engine

```
model.inference([input_ids, segment_ids, input_mask])  # input should be numpy array data
```

The `input_ids`, `segment_ids` and `input_mask` are the input numpy array data of a bert model, which have size (batch_size, seq_len). Note that the `out` is a dict contains the output tensor name and value(numpy array).

### 2. Manual customized yaml and weight binary to use Engine inference

### Build the yaml and weight binary

Engine could parse yaml structure network and load the weight from binary to do inference, yaml should like below

```
model:
  name: bert_model
  operator:
    input_data:
      type: Input                # define the input and weight shape/dtype/location
      output:
        input_ids:0:
          dtype: int32
          shape: [-1, -1]
        segment_ids:0:
          dtype: int32
          shape: [-1, -1]
        input_mask:0:
          dtype: int32
          shape: [-1, -1]
        bert/embeddings/word_embeddings:0:
          dtype: fp32
          shape: [30522, 1024]
          location: [0, 125018112]
          ....
    padding_sequence:                   # define the operators type/input/output/attr
      type: PaddingSequence
      input:
        input_mask:0: {}
      output:
        padding_sequence:0: {}
      attr:
        dst_shape: -1,16,0,-1
        dims: 1
    bert/embeddings/Reshape:
      type: Reshape
      input:
        input_ids:0: {}
      output:
        bert/embeddings/Reshape:0: {}
      attr:
        dst_shape: -1
    ....
    output_data:                       # define the output tensor
      type: Output
      input:
        logits:0: {}

```
All input tensors are in an operator typed Input. But slightly difference is some tensors have location while others not. A tensor with location means that is a frozen tensor or weight, it's read from the bin file. A tensor without location means it's activation, that should feed to the model during inference.

### Run the inference by Engine

Parse the yaml and weight bin to Engine Graph

```
from engine.compile.graph import Graph
model = Graph()
model.graph_init('./ir/conf.yaml', './ir/model.bin')
input_data = [input_0, input_1, input_2]
out = model.inference(input_data)
```

Same as the previous session, the ***input_data*** should be numpy array data as a list, and ***out*** is a dict the output tensor name and value(numpy array).

If you want to close log information during inference, use the command `export GLOG_minloglevel=2` before run the inference to set log level to ERROR.  `export GLOG_minloglevel=1` set log level to info again.
