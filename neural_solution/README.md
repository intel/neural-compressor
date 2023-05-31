# What's Neural Solution?

Neural Solution is a flexible and easy to use tool that brings the capabilities of Intel® Neural Compressor as a service. Users can effortlessly submit optimization tasks through the RESTful/gRPC APIs. Neural Solution automatically dispatches these tasks to one or multiple nodes, streamlining the entire process.

# Why Neural Solution?

- Efficiency: Neural Solution accelerates the optimization process by seamlessly parallelizing the tuning across multiple nodes.
- APIs: REST and gRPC are supported for submitting optimization tasks.
- Code Less: When working with Hugging Face models, Neural Solution drives the optimization process without requiring any code modifications by integrating the Neural Coder's functionality.

# How does Nueral Solution Work?
![NS-OaaS-Intro](./docs/source/imgs/NS-OaaS-Intro.png)

# Get Started
## Installation
<details>
  <summary>Prerequisites</summary>

<!-- TODO updated it by following  https://pytorch.org/get-started/locally/#windows-prerequisites-2 -->
- A working MPI implementation
- Python: >= 3.8
- Conda: >= 4.10.3
</details>

### Use pip to install neural solution:
```
pip install neural-solution
```

### Build from source:
```
python setup.py neural_solution install
```


# E2E examples
- [Quantizing a Hugging Face model](./examples/hf_models/README.md)
- [Quantizing a custom model](./examples/custom_models_optimized/tf_example1/README.md)
# Learn More
<!-- TODO more docs(Install details, API and so on...) -->

- The Architecture documents
- [APIs Reference](./docs/source/description_api.md)

# Contact

Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Solution related question.


