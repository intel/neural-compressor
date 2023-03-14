User Guide
===========================

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search to help the user optimize their model. The below documents could help you to get familiar with concepts and modules in Intel® Neural Compressor. Learn how to utilize the APIs in Intel® Neural Compressor to conduct quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks.

## Documentation

* **Overview** part helps user to get a quick understand about design structure and workflow of Intel® Neural Compressor. We provided broad examples to help users get started.   
* **Python-based APIs** contains more details about the functional APIs in Intel® Neural Compressor, which introduce the mechanism of each function and provides a tutorial to help the user apply in their own cases. Please note that we will stop to support Intel Neural Compressor 1.X API in the future. So we provide a comprehensive migration document in **Code Migration** to help the user update their code from previous 1.X version to the new 2.X version.   
* **Neural Coder** shows our special innovation about zero-code optimization to help user quickly apply Intel® Neural Compressor optimization without coding.  
* **Advanced Topics** provide the advanced topics that help user dive deep into Intel® Neural Compressor.  

<table class="docutils">
  <thead>
  <tr>
    <th colspan="9">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="4" align="center"><a href="design.md#architecture">Architecture</a></td>
      <td colspan="3" align="center"><a href="design.md#workflow">Workflow</a></td>
      <td colspan="1" align="center"><a href="api-doc/apis.html">APIs</a></td>
      <td colspan="1" align="center"><a href="bench.md">GUI</a></td>
    </tr>
    <tr>
      <td colspan="2" align="center"><a href="/examples/README.md#notebook-examples">Notebook</a></td>
      <td colspan="1" align="center"><a href="/examples/README.md">Examples</a></td>
      <td colspan="1" align="center"><a href="validated_model_list.md">Results</a></td>
      <td colspan="5" align="center"><a href="https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html">Intel oneAPI AI Analytics Toolkit</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Python-based APIs</th>
    </tr>
  </thead>
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
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Code Migration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="9" align="center"><a href="migration.md">Code Migration from Intel® Neural Compressor 1.X to Intel® Neural Compressor 2.X</a></td>
    </tr>    
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Neural Coder (Zero-code Optimization)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="1" align="center"><a href="neural_coder/docs/PythonLauncher.md">Launcher</a></td>
        <td colspan="2" align="center"><a href="neural_coder/extensions/neural_compressor_ext_lab/README.md">JupyterLab Extension</a></td>
        <td colspan="3" align="center"><a href="neural_coder/extensions/neural_compressor_ext_vscode/README.md">Visual Studio Code Extension</a></td>
        <td colspan="3" align="center"><a href="neural_coder/docs/SupportMatrix.md">Supported Matrix</a></td>
    </tr>    
  </tbody>
  <thead>
      <tr>
        <th colspan="9">Advanced Topics</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="3" align="center"><a href="adaptor.md">Adaptor</a></td>
          <td colspan="3" align="center"><a href="tuning_strategies.md">Strategy</a></td>
          <td colspan="3" align="center"><a href="distillation_quantization.md">Distillation for Quantization</a></td>
      </tr>
      <tr>
        <td colspan="3" align="center"><a href="metric.md">Metric</a></td>
        <td colspan="3" align="center"><a href="objective.md">Objective</a></td>
        <td colspan="3" align="center">SmoothQuant (Coming Soon)</td>
      </tr>
  </tbody>
</table>

