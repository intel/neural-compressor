# Calibration Algorithms in Quantization

1. [Introduction](#introduction)
2. [Calibration Algorithms](#calibration-algorithms)
3. [Support Matrix](#support-matrix)

## Introduction

Quantization proves beneficial in terms of reducing the memory and computational requirements of the model. Uniform quantization transforms the input value $x ∈ [β, α]$ to lie within $[−2^{b−1}, 2^{b−1} − 1]$, where $[β, α]$ is the range of real values chosen for quantization and $b$ is the bit-width of the signed integer representation. Calibration is the process of determining the $α$ and $β$ for model weights and activations. Refer to this [link](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md#quantization-fundamentals) for more quantization fundamentals

## Calibration Algorithms

Currently, Intel® Neural Compressor supports three popular calibration algorithms:

- MinMax: This method gets the maximum and minimum of input values as $α$ and $β$ [^1]. It preserves the entire range and is the simplest approach.

- Entropy: This method minimizes the KL divergence to reduce the information loss between full-precision and quantized data [^2]. Its primary focus is on preserving essential information.

- Percentile: This method only considers a specific percentage of values for calculating the range, ignoring the remainder which may contain outliers [^3]. It enhances resolution by excluding extreme values but still retaining noteworthy data.

## Support Matrix

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
    <td>OnnxRuntime</td>
    <td>minmax</td>
    <td>minmax, kl, percentile</td>
  </tr>
</tbody>
</table>

> `kl` is used to represent the Entropy calibration algorithm in Intel® Neural Compressor.

The calibration algorithm is one of the tuning items utilized by Intel® Neural Compressor auto-tuning. The accuracy-aware tuning process will select an appropriate algorithm. Please refer to [tuning_strategies.md](https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-space) for more details.

## Reference

[^1]: Vanhoucke, Vincent, Andrew Senior, and Mark Z. Mao. "Improving the speed of neural networks on CPUs." (2011).

[^2]: Szymon Migacz. "Nvidia 8-bit inference width tensorrt." (2017).

[^3]: McKinstry, Jeffrey L., et al. "Discovering low-precision networks close to full-precision networks for efficient embedded inference." arXiv preprint arXiv:1809.04191 (2018).
