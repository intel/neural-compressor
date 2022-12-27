TensorBoard
===========
1. [Introduction](#introduction)
2. [Supported Feature Matrix](#supported-feature-matrix)
3. [Get Started with Tensorboard](#get-started-with-tensorboard)
4. [Examples](#examples)

## Introduction

TensorBoard is a suite of web applications that provide measurements and visualizations used to inspect and understand your machine learning workflow for [TensorFlow TensorBoard](https://github.com/tensorflow/tensorboard) and [PyTorch TensorBoard](https://github.com/pytorch/pytorch/tree/master/torch/utils/tensorboard). Intel® Neural Compressor performs accuracy-driven quantization; the tuning process quantizes the tensor and performs graph transformation and optimization to achieve optimal performance under accuracy requirement. If you want to observe the behaviors of the optimizations, or if you want to discover why an accuracy target cannot be met, TensorBoard can provide you with some valuable information. You can inspect the graph and tensor after each tuning run. If a model cannot meet accuracy requirements, you can analyze the comparison of FP32 and the INT8 tensor histogram.    

We collect the TensorBoard event summary during evaluation. The first time is on the baseline FP32 model and later on at the end of each tuning runs are based on the quantized model. The TensorBoard log directory is named baseline_acc_<accuracy> and tune_<runs>_acc_<accuracy>, to indicate the stage and accuracy of the data that is generated. Users can select their data of interest to observe with TensorBoard. 

## Supported Feature Matrix
| Optimized Framework | Tensorboard Support |
|---------------------|:-------------------:|
| PyTorch             |       &#10004;      |
| TensorFlow          |       &#10004;      |

## Get Started with TensorBoard
### PyTorch TensorBoard

PyTorch TensorBoard implementation includes three steps:

* Before evaluation in the _pre_eval_hook() where instruments observers are placed in the model.
* During evaluation where observers collect tensor information in a dict data structure.
* After evaluation where the graph and tensor information is dumped with the TensorBoard summary writer in _post_eval_hook().


The detailed algorithm can be described by the Pseudo code:
```python

def evaluate(self, model, dataloader, postprocess=None, \
                 metric=None, measurer=None, iteration=-1, tensorboard=False):
# The tensorboard summary is collected in the evaluation function of adaptor

    if tensorboard:
         model = self._pre_eval_hook(model) 
    #evaluation code
    ....
    acc = metric.result()     
    if tensorboard: 
         self._post_eval_hook(model, accuracy=acc, input=input) 

def _pre_eval_hook(self, model):
# Insert observer submodule into each module in whitelist in order to collect tensor information

   class _RecordingObserver(ABC, torch.nn.Module):
   # Define the Observer class 

        def forward(self, x):
        # Record the tensor information in a dict structure
            self.output_tensors_dict[self.current_iter] = x.to("cpu") 

        @torch.jit.export
        def get_tensor_value(self):
            return self.output_tensors_dict

   def _observer_forward_hook(module, input, output):
        #Forward hook that calls observer on the output
        return module.activation_post_process(output)

   def _add_observer_(module, op_list=None, prefix=""): 

        #Add observer for each child module
        for name, child in module.named_children():
            _add_observer_(child, op_list, op_name)

        if module is a leaf:
           module.add_module(
                    'activation_post_process',
                    module.qconfig.activation())
                module.register_forward_hook(_observer_forward_hook)

def _post_eval_hook(self, model, **args):
   # Dump tensor and graph information with TensorBoard summary writer
    if self.dump_times == 0:
       writer = SummaryWriter('runs/eval/baseline' +
                             '_acc' + str(accuracy), model)
    else:
       writer = SummaryWriter('runs/eval/tune_' +
                                  str(self.dump_times) +
                                  '_acc' + str(accuracy), model)

    if args is not None and 'input' in args and self.dump_times == 0:
       writer.add_graph(model, args['input'])

    from torch.quantization import get_observer_dict
    get_observer_dict(model, observer_dict)
    for key in observer_dict:
        ......
        op_name = key.strip(".activation_post_process")
        summary[op_name + ".output"] = observer_dict[key].get_tensor_value()
        
        for iter in summary[op_name + ".output"]:
            #Record output tensor, for fused op only record the parent op output 
            ......
            if summary[op_name + ".output"][iter].is_quantized:
                  writer.add_histogram(
                        op + "/Output/int8",
                        torch.dequantize(summary[op_name +
                                                 ".output"][iter]))
            else:
                  writer.add_histogram(
                        op + "/Output/fp32",
                        summary[op_name + ".output"][iter])

        state_dict = model.state_dict()
        for key in state_dict:
            # Record weight tensor, fused child tensorBoard tag will be merge 
            if state_dict[key].is_quantized:
                writer.add_histogram(op + "/int8",
                                     torch.dequantize(state_dict[key]))
            else:
                writer.add_histogram(op + "/fp32", state_dict[key])
      
```
 

#### Usage

1. Add "tensorboard: true" in the yaml file.
2. Run quantization tuning; a "./runs" folder is generated in the working folder.
3. Start tensorboard:

   ``shell
     tensorboard --bind_all --logdir_spec baseline:./runs/eval/tune_0_acc0.80,tune_1:././runs/eval/tune_1_acc0.79  
   ``


### TensorFlow Tensorboard

TensorFlow TensorBoard implementation includes four steps:

1. Before evaluation where we create the TensorBoard summary write and write graph, collect FP32 and node names for inspection, and dump the histogram of weights and bias tensor directly from graph_def.
2. Run get_tensor_by_name_with_import() where we get data output tensors.
3. Run session.run() to predict and get the inference result of the output tensor list collected in the previous step.
4. Enumerate the output tensor and write the histogram.   

See the [tensorflow.py](https://github.com/intel/neural-compressor/tree/master/neural_compressor/adaptor/tensorflow.py) evaluate() function for details. 

#### Usage

1. Add "tensorboard: true" in the yaml file.

2. Run quantization tuning; a "./runs" folder is generated in the working folder. For example: 

   ```shell
   ls ./runs/eval  
   baseline_acc_0.776  tune_1_acc_0.095 
   ```
   The baseline_acc_0.776 folder contains the FP32 event log and 0.776 is the FP32 accuracy. tune_1_acc_0.095 contains the evaluation event log of the first run of tuning.  

3. Start tensorboard:

   ```shell
   tensorboard --bind_all --logdir_spec baseline:./runs_v3/eval/baseline_acc_0.776/,tune_1:./runs_v3/eval/tune_1_acc_0.095/ 
   ```
## Examples

### PyTorch Examples

```shell
  examples/pytorch/eager/image_recognition/imagenet/cpu/ptq/run_tuning_dump_tensor.sh 
```

### TensorFlow Examples

1. Add "tensorboard: true" into examples/tensorflow/image_recognition/inceptionv3.yaml. In order to demonstrate the usage of TensorBoard, remove the following lines which are added to skip the quantization of 'v0/cg/conv0/conv2d/Conv2D' to avoid a known limitation.

   ```yaml
      op_wise: {
               'v0/cg/conv0/conv2d/Conv2D': {
                 'activation':  {'dtype': ['fp32']},
               }
             }
   ```

2. Run tuning:

   ```shell
   bash run_tuning.sh --topology=inception_v3 --dataset_location=<imagenet> \
            --input_model=./inceptionv3_fp32_pretrained_model.pb --output_model=./nc_inceptionv3.pb  --config=./inceptionv3_dump_tensor.yaml 
   ```

3. Start TensorBoard

   ```shell
   tensorboard --bind_all --logdir_spec baseline:./runs_v3/eval/baseline_acc_0.776/,tune_1:./runs_v3/eval/tune_1_acc_0.095/
   ```

4. In order to find the reason why tune_1 got such poor accuracy, we can observe the TensorBoard.

* From the **GRAPHS** tab, select "baseline/." in the "Run" box and find the first 'Conv2d' op after 'input' op. The op name is "v0/cg/conv0/Relu":

![TensorBoard Baseline](./_static/imgs/tensorboard_baseline_v0_cg_conv0.png "TensorBoard Baseline")

* From the **GRAPHS** tab, select "tune_1/." in the "Run" box and find the first 'Conv2d' op after 'input' op. The tensor name is 'v0/cg/conv0/conv2d/Conv2D_eightbit_requantize':

![TensorBoard Tuning](./_static/imgs/tensorboard_tune_1_v0_cg_conv0.png "TensorBoard Tuning")


* Switch to the **HISTOGRAMS** tab. Click the 'v0/cg/conv0' op name in the search box. TensorBoard groups the tensors with the same op name together so you can compare the tensor of baseline 'v0/cg/conv0/Relu' with the tensor of tune_1 'v0/cg/conv0/conv2d/Conv2D_eightbit_requantize_int8.output'. Note that the tensor name can be changed after quantization, so group the tensor by op name and compare. From the chart below, we can see that the histogram of the first conv2d output tensor are different. This is due to a known TensorFlow issue. After filtering the 'v0/cg/conv0/conv2d/Conv2D' op by adding "op_wise" in the yaml file, the issue disappears.  
 
![TensorBoard Histogram](./_static/imgs/tensorboard_v0_cg_conv0_histogram.png "TensorBoard Histogram")
