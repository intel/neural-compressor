### Tensorflow

Intel has worked with the TensorFlow development team to enhance TensorFlow to include bfloat16 data support for CPUs. For more information about BF16 in TensorFlow, please read [Accelerating AI performance on 3rd Gen Intel® Xeon® Scalable processors with TensorFlow and Bfloat16](https://blog.tensorflow.org/2020/06/accelerating-ai-performance-on-3rd-gen-processors-with-tensorflow-bfloat16.html).

- BF16 conversion during quantization in TensorFlow

![Mixed Precision](imgs/bf16_convert_tf.png "Mixed Precision Graph")

- Three steps

1. Convert to a `FP32 + INT8` mixed precision Graph

   In this steps, TF adaptor will regard all fallback datatype as `FP32`. According to the per op datatype in tuning config passed by strategy, TF adaptor will generate a `FP32 + INT8` mixed precision graph.

2. Convert to a `BF16 + FP32 + INT8` mixed precision Graph

   In this phase, adaptor will convert some `FP32` ops to `BF16` according to `bf16_ops` list in tuning config.

3. Optimize the `BF16 + FP32 + INT8` mixed precision Graph
   
   After the mixed precision graph generated, there are still some optimization need to be applied to improved the performance, for example `Cast + Cast` and so on. The `BF16Convert` transformer also apply a depth-first method to make it possible to take the ops use `BF16` which can support `BF16` datatype to reduce the insertion of `Cast` op.

### PyTorch

Intel has also worked with the PyTorch development team to enhance PyTorch to include bfloat16 data support for CPUs.

- BF16 conversion during quantization in PyTorch

![Mixed Precision](imgs/bf16_convert_pt.png "Mixed Precision Graph")

- Two steps
1. Convert to a `FP32 + INT8` mixed precision Graph or Module

   In this steps, PT adaptor will combine the `INT8` ops and all fallback ops to `FP32 + INT8` mixed precision Graph or Module no matter in Eager mode or Fx Graph mode.

2. Convert to a `BF16 + FP32 + INT8` mixed precision Graph or Module

   In this phase, adaptor will according to `BF16` op list from strategy tune config to wrapper the `FP32` module with `BF16Wrapper` to realize the `BF16 + FP32 + INT8` mixed precision Graph or Module. adaptor will do retrace the `GraphModule` again if using Fx Graph mode.