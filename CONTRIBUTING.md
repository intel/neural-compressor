# Contributing guidelines
If you have improvements to Intel® Low Precision Optimization Tool, send us your pull requests for
[review](https://github.com/intel/lpot/pulls)! For those
just getting started, Github has a
[how to](https://help.github.com/articles/using-pull-requests/).

## Pull Request Checklist
Before sending your pull requests, please make sure that you followed this
list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Changes are consistent with the Python [Coding Style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
- Use pylint to check your Python code
- Use flake8 and autopep8 to make Python code clean 
- Add unit tests in [Unit Tests](https://github.com/intel/lpot/tree/master/test) to cover the code you would like to contribute.
- Run [Unit Tests](https://github.com/intel/lpot/tree/master/test).

## Pull Request Template
### Change Summary
Include a detailed summary of the change please. 

### Change Motivation
Include an explanation of the motivation for the change please.

### Change Limit
Include an explanation about the regression your change might bring.

### Test Info
- For bug fix, provide test steps to reproduce your issue. 
- For new features, provide test steps besides unit tests if necessary. 

### Environment Info
Privide the development or test environment info.
- OS
- CPU info
- Python version
- Dependent component version

