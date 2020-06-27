Introduction
=========================================

Intel Low Precision Tool (iLiT) is an accuracy-driven auto tuning library for low precision application cross DL frameworks.

# Infrastructure

<div align="center">
  <img src="imgs/infrastructure.jpg" width="700px" />
  <p>iLiT Infrastructure</p>
</div>

# Workflow

<div align="center">
  <img src="imgs/workflow.jpg" width="700px" />
  <p>iLiT Workflow</p>
</div>

# Strategies

### Basic Strategy

This strategy is iLiT default tuning strategy, which does model-wise tuning by adjusting gloabl tuning parameters, such as calibration related parameters, kl or minmax algo, quantization related parameters, symmetric or asymmetric, per_channel or per_tensor. If the model-wise tuning result doesn't meet accuracy goal, this strategy will attempt to do op-wise fallback from bottom to top to prioritize which fallback op has biggest impact on final accuracy, and then do incremental fallback till achieving the accuracy goal.

### Random Strategy

This strategy is used to random choose tuning config from tuning space.

### Exhaustive Strategy

This strategy is used to sequentially traverse all the possible tuning configs in tuning space.

### Bayersian Strategy

TODO

### MSE Strategy

TODO

# Objectives

iLiT supports below 3 build-in objectives:

### 1. Performance

This objective is used to measure performance of quantized model with accuracy driven. It's default objective iLiT supported.

### 2. Memory Footprint

This objective is used to measure the memory usage of evaluating quantized model with accuracy driven.

### 3. Model Size

This objective is used to measure the memory size of quantized model with accuracy driven.

# Metrics

iLiT supports 3 built-in metrics, Topk, F1 and CocoMAP. The metric is easily extensible as well as contributor implements a subclass of Metric class.
