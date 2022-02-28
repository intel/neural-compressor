# Benchmark an engine IR using C++ API
A deep learning inference engine for quantized and sparsified models.

## Architecture
Engine support model optimizer, model executor and high performance kernel for multi device.

<a target="_blank" href="imgs/architecture.png">
  <img src="imgs/architecture.png" alt="Architecture" width=762 height=672>
</a>

## Installation

Just support Linux operating system for now.


### 0.Prepare environment and requirement

```
# prepare your env
conda create -n <env name> python=3.7
conda install cmake --yes
conda install absl-py --yes
conda activate <env name>

# install tensorflow
pip install https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl

# install transformers
pip install transformers

# install INC
pip install neural-compressor
```

### 1.Clone the engine repo

`git clone https://github.com/intel-innersource/frameworks.ai.deep-engine.intel-deep-engine.git <work_folder>`

### 2.Generate the bert model intermediate representations, that are yaml and bin flies

```python
# import compile api form engine
from engine.compile import compile
# get the engine intermediate graph (input onnx or tf model)
graph = compile(<model_path>)
# save the graph and get the final ir
# the yaml and bin file will stored in '<ir_path>' folder
graph.save(<ir_path>)
```

Then in <ir_path>, you will see the corresponding yaml and bin files

### 3.Build the engine, **make sure your gcc version >= 7.3**

```
cd <work_folder>/engine/executor
mkdir build
cd build
cmake ..
make -j
```

Then in the build folder, you will get the `inferencer`, `engine_py.cpython-37m-x86_64-linux-gnu.so` and `libengine.so`. The first one is used for pure c++ model inference, and the second is used for python inference, they all need the `libengine.so`.

### 4.Use the `inferencer` for dummy/const data performance test

`./inferencer --config=<the generated yaml file path> --weight=<the generated bin file path> --batch_size=32 --iterations=20`

You can set the `batch_size` and `iterations` number. Besides you can use the `numactl` command to bind cpu cores and open multi-instances. For example:

`OMP_NUM_THREADS=4 numactl -C '0-3' ./inferencer --config=<the generated yaml file path> --weight=<the generated bin file path> --batch_size=32 --iterations=20`

Then, you can see throughput result of the `inferencer`  on the terminal.

please remember to change the input data type, shape and range for your input in `inferencer.cpp`

```
  vector<string> input_dtype(3, "int32");
  vector<vector<float>> input_range(3, vector<float>({1, 300}));
  vector<vector<int64_t>> input_shape(3, {FLAGS_batch_size, FLAGS_seq_len});
  executor::DataLoader* dataloader;
  // dataloader = new executor::ConstDataLoader(input_shape, input_dtype, input_range);
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);

```
The dataloader generate data using prepare_batch() and make sure the generate data is the yaml need:

```
model:
  name: bert_mlperf_int8
  operator:
    input_data:
      type: Input
      output:
        # -1 means it's dynamic shape

        input_ids:
          dtype: int32
          shape: [-1, -1]
        segment_ids:
          dtype: int32
          shape: [-1, -1]
        input_mask:
          dtype: int32
          shape: [-1, -1]
```
All input tensors are in an operator typed Input. But slightly difference is some tensors have location while others not. A tensor with location means that is a frozen tensor or weight, it's read from the bin file. A tensor without location means it's activation, that should be input during model Forward. When you use C++ interface, initialize the tensor config and feed data/shape from dataloader:

```
  // initialize the input tensor config(which correspond to the yaml tensor without location)
  vector<executor::Tensor> input_tensors;
  auto input_configs = bert_model.input_configs();
  for (int i = 0; i < bert_model.num_inputs(); ++i) {
    input_tensors.push_back(executor::Tensor(*(input_configs[i])));
  // feed the data and shape
  auto raw_data = dataloader->prepare_batch(i);
  for (int i = 0; i < input_tensors.size(); ++i) {
    input_tensors[i].set_data(raw_data[i]);
    input_tensors[i].set_shape(input_shape[i]);
  }

```

The output tensor is defined in an operator named Output, which only have inputs as follows:

```
  output_data:
    type: Output
    input:
      matmul_post_output_reshape: {}

```
You can add the tensor you want to the Output. Remember, all output tensors from operators should have operators take as input, that means output edges should have an end node. In this case, each operator's output has one or several other operators take as input.

If you want to close log information of the `inferencer`, use the command `export GLOG_minloglevel=2` before executing the `inferencer`.  `export GLOG_minloglevel=1` will open the log information again. This command can also be used in python engine model.
