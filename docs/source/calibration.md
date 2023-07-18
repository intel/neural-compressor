# Calibration Algorithm in Quantization

1. [Introduction](#introduction)
2. [Calibration Algorithm](#calibration-algorithm)
3. [Supported Framework Matrix](#supported-framework-matrix)

## Introduction

Quantization proves beneficial in terms of reducing the memory and computational requirements of the model. Uniform quantization transforms the input value $x ∈ [β, α]$ to lie within $[−2^{b−1}, 2^{b−1} − 1]$, where $[β, α]$ is the range of real values chosen for quantization and $b$ is the bit-width of the signed integer representation. Calibration is the process of determining the $α$ and $β$ for model weights and activations. Refer to this [link](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md#quantization-fundamentals) for more quantization fundamentals

## Calibration Algorithm

There are three main methods of Calibration:

- MinMax: This is the simplest method to get the maximum and minimum of input values as $α$ and $β$ [^1]. This approach maintains the full range but often at the cost of compromising precision.

- Entropy: This method minimizes the KL divergence to reduce the information loss between the  full-precision and the quantized data [^2]. By focusing on preserving essential information, this approach can better manage precision.

- Percentile: This approach only considers a specific percentage of values for calculating the range, ignoring the remainder which may contain outliers [^3]. This method enhances resolution by excluding extreme values but still retaining noteworthy data.


## Supported Framework Matrix

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th colspan="2">Supported calibration algorithm</th>
  </tr>
  <tr>
    <th>weight</th>
    <th>activation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Pytorch</td>
    <td>minmax</td>
    <td>minmax, kl</td>
  </tr>
  <tr>
    <td>Tensorflow</td>
    <td>minmax</td>
    <td>minmax, kl</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>minmax</td>
    <td>minmax, kl</td>
  </tr>
  <tr>
    <td>OnnxRuntime</td>
    <td>minmax</td>
    <td>minmax, kl, percentile</td>
  </tr>
</tbody>
</table>

> `kl` is used to represent the Entropy calibration algorithm in Intel® Neural Compressor.

Calibration algorithm is in the tuning space of Intel® Neural Compressor. After tuning, a quantization model with the most suitable calibration algorithm will be returned. Please refer to [tuning_strategies.md](https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-space) for more details.

## Reference

[^1]: Vanhoucke, Vincent, Andrew Senior, and Mark Z. Mao. "Improving the speed of neural networks on CPUs." (2011).

[^2]: Szymon Migacz. "Nvidia 8-bit inference width tensorrt." (2017).

[^3]: McKinstry, Jeffrey L., et al. "Discovering low-precision networks close to full-precision networks for efficient embedded inference." arXiv preprint arXiv:1809.04191 (2018).
