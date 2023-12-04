User Guide
===========================

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search to help the user optimize their model. The below documents could help you to get familiar with concepts and modules in Intel® Neural Compressor. Learn how to utilize the APIs in Intel® Neural Compressor to conduct quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks.

## Overview
This part helps user to get a quick understand about design structure and workflow of Intel® Neural Compressor. We provided broad examples to help users get started.
<table class="docutils">
<tbody>
<tr>
  <td colspan="4" align="center"><a href="design.md#architecture">Architecture</a></td>
  <td colspan="3" align="center"><a href="design.md#workflow">Workflow</a></td>
  <td colspan="2" align="center"><a href="https://intel.github.io/neural-compressor/latest/docs/source/api-doc/apis.html">APIs</a></td>
</tr>
<tr>
  <td colspan="2" align="center"><a href="/examples/README.md#notebook-examples">Notebook</a></td>
  <td colspan="1" align="center"><a href="/examples/README.md">Examples</a></td>
  <td colspan="1" align="center"><a href="validated_model_list.md">Results</a></td>
  <td colspan="5" align="center"><a href="https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html">Intel oneAPI AI Analytics Toolkit</a></td>
</tr>
</tbody>
</table>

## Python-based APIs
Python-based APIs contains more details about the functional APIs in Intel® Neural Compressor,
which introduce the mechanism of each function and provides a tutorial to help the user apply in their own cases.
Please note that we will stop to support Intel Neural Compressor 1.X API in the future.
So we provide a comprehensive migration document in Code Migration to help the user update their code from previous 1.X version to the new 2.X version.  
In 2.X API, it's very important to create the `DataLoader` and `Metrics` for your examples, so we provide the detail introductions.

<table class="docutils">
<tbody>
<tr>
<td colspan="2" align="center"><a href="quantization.md">Quantization</a></td>
<td colspan="3" align="center"><a href="mixed_precision.md">Advanced Mixed Precision</a></td>
<td colspan="2" align="center"><a href="pruning.md">Pruning (Sparsity)</a></td>
<td colspan="2" align="center"><a href="distillation.md">Distillation</a></td>
</tr>
<tr>
<td colspan="2" align="center"><a href="orchestration.md">Orchestration</a></td>
<td colspan="2" align="center"><a href="benchmark.md">Benchmarking</a></td>
<td colspan="3" align="center"><a href="distributed.md">Distributed Compression</a></td>
<td colspan="3" align="center"><a href="export.md">Model Export</a></td>
</tr>
<tr>
<td colspan="9" align="center"><a href="migration.md">Code Migration from Intel® Neural Compressor 1.X to Intel® Neural Compressor 2.X</a></td>
</tr>
<tr>
<td colspan="4" align="center"><a href="dataloader.md">DataLoader</a></td>
<td colspan="5" align="center"><a href="metric.md">Metric</a></td>
</tr>
</tbody>
</table>

## Neural Coder (Zero-code Optimization)
Neural Coder shows our special innovation about zero-code optimization to help user quickly apply Intel® Neural Compressor optimization without coding.
<table class="docutils">
<tbody>
<tr>
<td colspan="1" align="center"><a href="/neural_coder/docs/PythonLauncher.md">Launcher</a></td>
<td colspan="2" align="center"><a href="/neural_coder/extensions/neural_compressor_ext_lab/README.md">JupyterLab Extension</a></td>
<td colspan="3" align="center"><a href="/neural_coder/extensions/neural_compressor_ext_vscode/README.md">Visual Studio Code Extension</a></td>
<td colspan="3" align="center"><a href="/neural_coder/docs/SupportMatrix.md">Supported Matrix</a></td>
</tr>
</tbody>
</table>

## Advanced Topics
This part provides the advanced topics that help user dive deep into Intel® Neural Compressor.
<table class="docutils">
<tbody>
<tr>
<td colspan="3" align="center"><a href="adaptor.md">Adaptor</a></td>
<td colspan="3" align="center"><a href="tuning_strategies.md">Strategy</a></td>
<td colspan="3" align="center"><a href="objective.md">Objective</a></td>
<td colspan="3" align="center"><a href="calibration.md">Calibration</a></td>
</tr>
<tr>
<td colspan="4" align="center"><a href="diagnosis.md">Diagnosis</a></td>
<td colspan="4" align="center"><a href="add_new_data_type.md">Add New Data Type</a></td>
<td colspan="4" align="center"><a href="add_new_adaptor.md">Add New Adaptor</a></td>
</tr>
<tr>
<td colspan="4" align="center"><a href="distillation_quantization.md">Distillation for Quantization</a></td>
<td colspan="4" align="center"><a href="smooth_quant.md">SmoothQuant</a></td>
<td colspan="4" align="center"><a href="quantization_weight_only.md">Weight-Only Quantization</a></td>
</tr>
</tbody>
</table>

## Innovations for Productivity
We are continue creating some user-friendly applications to improve the productivity. From v2.2 we have `Neural Solution` for distributed quantization and `Neural Insights` for quantization accuracy debugging.
<table class="docutils">
<tbody>
<tr>
<td colspan="4" align="center"><a href="/neural_solution/README.md">Neural Solution</a></td>
<td colspan="4" align="center"><a href="/neural_insights/README.md">Neural Insights</a></td>
</tr>
</tbody>
</table>
