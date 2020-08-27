BF16 Convert
=========================================

The recent growth of Deep Learning has driven the development of more complex models that require significantly more compute and memory capabilities. Several low precision numeric formats have been proposed to address the problem. Google's [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and the [FP16: IEEE](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) half-precision format are two of the most widely used sixteen bit formats. [Mixed precision](https://arxiv.org/abs/1710.03740) training and inference using low precision formats have been developed to reduce compute and bandwidth requirements.

The recently launched 3rd Gen Intel® Xeon® Scalable processor (codenamed Cooper Lake), featuring Intel® Deep Learning Boost, is the first general-purpose x86 CPU to support the bfloat16 format. Specifically, three new bfloat16 instructions are added as a part of the AVX512_BF16 extension within Intel Deep Learning Boost: VCVTNE2PS2BF16, VCVTNEPS2BF16, and VDPBF16PS. The first two instructions allow converting to and from bfloat16 data type, while the last one performs a dot product of bfloat16 pairs. Further details can be found in the [hardware numerics document](https://software.intel.com/content/www/us/en/develop/download/bfloat16-hardware-numerics-definition.html) published by Intel.

Intel has worked with the TensorFlow development team to enhance TensorFlow to include bfloat16 data support for CPUs. For more information about BF16 in TensorFlow, please read [Accelerating AI performance on 3rd Gen Intel® Xeon® Scalable processors with TensorFlow and Bfloat16](https://blog.tensorflow.org/2020/06/accelerating-ai-performance-on-3rd-gen-processors-with-tensorflow-bfloat16.html).

Intel® Low Precision Optimization Tool can support op-wise BF16 precision for TensorFlow now. With BF16 support, it can get a mixed precision model with acceptable accuracy and performance or others objective goals. This document will give a simple introduction of TensorFlow BF16 convert transformation and how to use the BF16.

# BF16 Convert Transformation in TensorFlow

<div align="left">
  <img src="imgs/bf16_convert_tf.png" width="700px" />
</div>

### Three steps

1. Convert to a `FP32 + INT8` mixed precision Graph

   In this steps, TF adaptor will regard all fallback datatype as `FP32`. According to the per op datatype in tuning config passed by strategy, TF adaptor will generate a `FP32 + INT8` mixed precision graph.

2. Convert to a `BF16 + FP32 + INT8` mixed precision Graph

   In this phase, adaptor will convert some `FP32` ops to `BF16` according to `bf16_ops` list in tuning config.

3. Optimize the `BF16 + FP32 + INT8` mixed precision Graph
   
   After the mixed precision graph generated, there are still some optimization need to be applied to improved the performance, for example `Cast + Cast` and so on. The `BF16Convert` transformer also apply a depth-first method to make it possible to take the ops use `BF16` which can support `BF16` datatype to reduce the insertion of `Cast` op.

# How to use it

> BF16 Convert in TensorFlow, it is a relatively new feature. And its enable requires the cooperation of software and hardware. For hardware, it need the CPU support `avx512_bf16` instruction set. And for software, it needs the `intel-tensorflow` has support `BF16` with oneDNN backend. We also support force enable it by using set the environment variable `FORCE_BF16=1`. But without above 2 sides support, the poor performance or other problems may occur.

 For now, if we want try this feature, we can follow below steps.

1. Build TensorFlow by using below command, we test this feature on TensorFlow master commit [f0b33d6](https://github.com/tensorflow/tensorflow/tree/f0b33d6feea2044ac0f9ccdd67f19ebc85adaab2)
```shell
bazel build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --copt=-O3 --copt=-Wformat --copt=-Wformat-security \
        --copt=-fstack-protector --copt=-fPIC --copt=-fpic --linkopt=-znoexecstack --linkopt=-zrelro \
        --linkopt=-znow --linkopt=-fstack-protector --config=mkl --define build_with_mkl_dnn_v1_only=true \
        --copt=-DENABLE_INTEL_MKL_BFLOAT16 --copt=-march=native //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
```

2. Install the wheel into your test python environment.

3. Add `bf16` in `weight` and `activation dtype` of `yaml` config file . By default, it has been added.
