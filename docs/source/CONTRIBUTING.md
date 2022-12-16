Contribution Guidelines
=======================

1. [Pull Request Checklist](#pull-request-checklist)
2. [Pull Request Template](#distillation-support-matrix)
3. [Support](#support)
4. [Contributor Covenant Code of Conduct](#contributor-covenant-code-of-conduct)

If you have improvements to Intel® Neural Compressor, send your pull requests for
[review](https://github.com/intel/neural-compressor/pulls). If you are new to Github, view the pull request [How To](https://help.github.com/articles/using-pull-requests/).

## Pull Request Checklist

Before sending your pull requests, follow the information below:

- Changes are consistent with the Python [Coding Style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
- Use pylint to check your Python code.
- Use flake8 and autopep8 to make Python code clean.
- Add unit tests in [Unit Tests](https://github.com/intel/neural-compressor/tree/master/test) to cover the code you would like to contribute.
- Run [Unit Tests](https://github.com/intel/neural-compressor/tree/master/test).
- Intel® Neural Compressor has adopted the [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin), you must agree to the terms of Developer Certificate of Origin by signing off each of your commits.`Signed-off-by: Random J Developer <random@developer.example.org>`

## Pull Request Template

**Change Summary**

Include a detailed summary of the change.

**Change Motivation**

Include an explanation for the change.

**Change Limit**

Include an explanation about the regression your change might bring.

**Test Info**

- For bug fixes, provide test steps to reproduce your issue.
- For new features, provide test steps besides unit tests if necessary.

**Environment Info**
Provide the development or test environment info.

- OS
- CPU info
- Python version
- Dependent component version

## Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/intel/neural-compressor/issues) page. You may also reach out to [Maintainers](neural_compressor.maintainers@intel.com).

## Contributor Covenant Code of Conduct

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
