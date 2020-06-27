Introduction
=========================================

Intel Low Precision Tool (iLiT) is an accuracy-driven auto tuning library for low precision application

# Infrastructure

# Strategies

### Basic Strategy

This strategy is iLiT default tuning strategy, which does model-wise tuning by adjusting gloabl tuning parameters, such as calibration related parameters, kl or minmax algo, quantization related parameters, symmetric or asymmetric, per_channel or per_tensor. If the model-wise tuning result doesn't meet accuracy goal, this strategy will attempt to do op-wise fallback from bottom to top to prioritize which fallback op has biggest impact on final accuracy, and then do incremental fallback till achieving the accuracy goal.

### Random Strategy

This Strategy 

### Exhaustive Strategy

This

### Bayersian Strategy

This

### MSE Strategy

This

# Objectives

# Metrics