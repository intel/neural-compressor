Contribution Guidelines
=======================
1. [Create Pull Request](#create-pull-request)
2. [Pull Request Checklist](#pull-request-checklist)
3. [Pull Request Template](#pull-request-template)
4. [Pull Request Acceptance Criteria](#pull-request-acceptance-criteria)
5. [Pull Request Status Checks Overview](#pull-request-status-checks-overview)
6. [Support](#support)
7. [Contributor Covenant Code of Conduct](#contributor-covenant-code-of-conduct)

## Create Pull Request
If you have improvements to Intel® Neural Compressor, send your pull requests for
[review](https://github.com/intel/neural-compressor/pulls).
If you are new to GitHub, view the pull request [How To](https://help.github.com/articles/using-pull-requests/).
### Step-by-Step guidelines
- Star this repository using the button `Star` in the top right corner.
- Fork this Repository using the button `Fork` in the top right corner.
- Clone your forked repository to your pc.
`git clone "url to your repo"`
- Create a new branch for your modifications.
`git checkout -b new-branch`
- Add your files with `git add -A`, commit `git commit -s -m "This is my commit message"` and push `git push origin new-branch`.
- Create a [pull request](https://github.com/intel/neural-compressor/pulls).

## Pull Request Checklist

Before sending your pull requests, follow the information below:

- Changes are consistent with the [coding conventions](./coding_style.md).
- Add unit tests in [Unit Tests](https://github.com/intel/neural-compressor/tree/master/test) to cover the code you would like to contribute.
- Intel® Neural Compressor has adopted the [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin), you must agree to the terms of Developer Certificate of Origin by signing off each of your commits with `-s`, e.g. `git commit -s -m 'This is my commit message'`.

## Pull Request Template

See [PR template](/.github/pull_request_template.md)

## Pull Request Acceptance Criteria
- At least two approvals from reviewers

- All detected status checks pass

- All conversations solved

- Third-party dependency license compatible

## Pull Request Status Checks Overview
Intel® Neural Compressor use [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/pipelines/?view=azure-devops) for CI test.
And generally use [Azure Cloud Instance](https://azure.microsoft.com/en-us/pricing/purchase-options/pay-as-you-go) to deploy pipelines, e.g. Standard E16s v5.
|     Test Name                 |     Test Scope                                |     Test Pass Criteria    |
|-------------------------------|-----------------------------------------------|---------------------------|
|     Code Scan                 |     Bandit/CopyRight/DocStyle/SpellCheck       |     PASS          |
|     [DCO](https://github.com/apps/dco/)     |     Use `git commit -s` to sign off     |     PASS          |
|     Unit Test                 |     Pytest scripts under [test](/test)                |      PASS (No failure, No core dump, No segmentation fault, No coverage drop)      |
|     Model Test                |     Pytorch + TensorFlow + ONNX Runtime         |      PASS (Functionality pass, FP32/INT8 No performance regression)       |

## Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/intel/neural-compressor/issues) page. You may also reach out to [Maintainers](mailto:inc.maintainers@intel.com).

## Contributor Covenant Code of Conduct

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
