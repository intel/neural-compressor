Developer Documentation
#######################

Read the following material as you learn how to use Neural Compressor.

Get Started
===========

* `Transform <transform.md>`__ introduces how to utilize Neural Compressor's built-in data processing and how to develop a custom data processing method. 
* `Dataset <dataset.md>`__ introduces how to utilize Neural Compressor's built-in dataset and how to develop a custom dataset.
* `Metrics <metric.md>`__ introduces how to utilize Neural Compressor's built-in metrics and how to develop a custom metric.
* `UX <bench.md>`__ is a web-based system used to simplify Neural Compressor usage.
* `Intel oneAPI AI Analytics Toolkit Get Started Guide <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html>`__ explains the AI Kit components, installation and configuration guides, and instructions for building and running sample apps.
* `AI and Analytics Samples <https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics>`__ includes code samples for Intel oneAPI libraries.

.. toctree::
    :maxdepth: 1
    :hidden:

    transform.md
    dataset.md
    metric.md
    ux.md
    Intel oneAPI AI Analytics Toolkit Get Started Guide <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html>
    AI and Analytics Samples <https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics>


Deep Dive
=========

* `Quantization <Quantization.md>`__ are processes that enable inference and training by performing computations at low-precision data types, such as fixed-point integers. Neural Compressor supports Post-Training Quantization (`PTQ <PTQ.md>`__) and Quantization-Aware Training (`QAT <QAT.md>`__). Note that `Dynamic Quantization <dynamic_quantization.md>`__ currently has limited support.
* `Pruning <pruning.md>`__ provides a common method for introducing sparsity in weights and activations.
* `Benchmarking <benchmark.md>`__ introduces how to utilize the benchmark interface of Neural Compressor.
* `Mixed precision <mixed_precision.md>`__ introduces how to enable mixed precision, including BFP16 and int8 and FP32, on Intel platforms during tuning.
* `Graph Optimization <graph_optimization.md>`__ introduces how to enable graph optimization for FP32 and auto-mixed precision.
* `Model Conversion <model_conversion.md>` introduces how to convert TensorFlow QAT model to quantized model running on Intel platforms.
* `TensorBoard <tensorboard.md>`__ provides tensor histograms and execution graphs for tuning debugging purposes. 


.. toctree::
    :maxdepth: 1
    :hidden:

    Quantization.md
    PTQ.md
    QAT.md
    dynamic_quantization.md
    pruning.md
    benchmark.md
    mixed_precision.md
    graph_optimization.md
    model_conversion.md
    tensorboard.md
  
    
Advanced Topics
===============

* `Adaptor <adaptor.md>`__ is the interface between Neural Compressor and framework. The method to develop adaptor extension is introduced with ONNX Runtime as example. 
* `Tuning strategies <tuning_strategies.md>`__ can automatically optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria. The method to develop a new strategy is introduced.


.. toctree::
    :maxdepth: 1
    :hidden:

    adaptor.md
    tuning_strategies.md
