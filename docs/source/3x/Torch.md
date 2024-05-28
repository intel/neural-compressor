Torch
=================================================

1. [Introduction](#introduction)
2. [Torch API](#torch-api)
3. [Support matrix](#supported-matrix)
4. [Common Problem](#common-problem)


## Introduction

`neural_compressor.torch` provides a Torch-like API usage and integrates kinds of quantization methods.

## Torch API

### Quantization

### Autotune

### Save&Load

## Supported Matrix

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-9wq8">Method<br></th>
    <th class="tg-9wq8">Algorithm</th>
    <th class="tg-9wq8">Backend</th>
    <th class="tg-9wq8">Support Status</th>
    <th class="tg-9wq8">Usage Link</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">Weight Only Quantization<br></td>
    <td class="tg-9wq8">Round to Nearest (RTN)<br></td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">GPTQ (link to paper)<br></td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">AWQ (link to paper)</td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">AutoRound  (link to paper)</td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">TEQ (link to paper)</td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">HQQ (link to paper)</td>
    <td class="tg-9wq8">Torch eager (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Smooth Quantization</td>
    <td class="tg-9wq8">SmoothQuant (link to paper)</td>
    <td class="tg-9wq8">Intel-extension-for-pytorch (link to repo)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">Static Quantization</td>
    <td class="tg-9wq8" rowspan="2">Post-traning Static Quantization (link to torch document)</td>
    <td class="tg-9wq8">Intel-extension-for-pytorch (link to repo)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">torch.dynamo (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Dynamic Quantization</td>
    <td class="tg-9wq8">Post-traning Dynamic Quantization (link to torch document)</td>
    <td class="tg-9wq8">torch.dynamo (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Quantization&nbsp;&nbsp;&nbsp;Aware Training</td>
    <td class="tg-9wq8">Quantization Aware Training (link to torch document)</td>
    <td class="tg-9wq8">torch.dynamo (link to torch document)</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
</tbody></table>

## Common Problem

1.  How to choose backend between `intel-extension-for-pytorch` and `torch.dynamo`?
    > Neural Compressor provides automatic logic to detect which backend should be used.
    > <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
    </style>
    <table class="tg"><thead>
    <tr>
        <th class="tg-9wq8">Environment</th>
        <th class="tg-9wq8">Automatic Backend</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-9wq8">import torch</td>
        <td class="tg-9wq8">torch.dynamo</td>
    </tr>
    <tr>
        <td class="tg-9wq8">import torch<br>import intel-extension-for-pytorch</td>
        <td class="tg-9wq8">intel-extension-for-pytorch</td>
    </tr>
    </tbody>
    </table>
