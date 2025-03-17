Release
=======

1. [Release Notes](#release-notes)
2. [Known Issues](#known-issues)
3. [Incompatible Changes](#incompatible-changes)

## Release Notes

View new feature information and release downloads for the latest and previous releases on GitHub. Validated configurations and distribution sites located here as well:

> <https://github.com/intel/neural-compressor/releases>

Contact [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) if you need additional assistance.

## Known Issues

The MSE tuning strategy does not work with the PyTorch adaptor layer. This strategy requires a comparison between the FP32 and INT8 tensors to decide which op impacts the final quantization accuracy. The PyTorch adaptor layer does not implement this inspect tensor interface. Therefore, do not choose the MSE tuning strategy for PyTorch models.

## Incompatible Changes

[Neural Compressor v1.2](https://github.com/intel/neural-compressor/tree/v1.2) introduces incompatible changes in user facing APIs. Please refer to [incompatible changes](incompatible_changes.md) to know which incompatible changes are made in v1.2.

[Neural Compressor v1.2.1](https://github.com/intel/neural-compressor/tree/v1.2.1) solves this backward compatible issues introduced in v1.2 by moving new user facing APIs to neural_compressor.experimental package and keep old one as is. Please refer to [API documentation](./api-documentation/apis.rst) to know the details of user-facing APIs.

[Neural Compressor v1.7](https://github.com/intel/neural-compressor/tree/v1.7) renames the pip/conda package name from lpot to neural_compressor. To run old examples on latest software, please replace package name for compatibility with `sed -i "s|lpot|neural_compressor|g" your_script.py` .

[Neural Compressor v2.0](https://github.com/intel/neural-compressor/tree/v2.0) renames the `DATASETS` class as `Datasets`, please notice use cases like `from neural_compressor.data import Datasets`. Details please check the [PR](https://github.com/intel/neural-compressor/pull/244/files).

[Neural Compressor v2.2](https://github.com/intel/neural-compressor/tree/v2.2) from this release, binary `neural-compressor-full` is deprecated, we deliver 3 binaries named `neural-compressor`, `neural-solution` and `neural-insights`.
