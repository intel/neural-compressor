# What's Neural Solution?
<!-- TODO what is neural_solution -->
Neural Solution is a flexible and easy to use tool that brings the capabilities of Intel® Neural Compressor as a service. Users can effortlessly submit optimization tasks through the RESTful/gRPC APIs. Neural Solution automatically dispatches these tasks to one or multiple nodes, streamlining the entire process.

# Why Neural Solution?
<!-- TODO what does the neural_solution provide -->
- Efficiency: Neural Solution accelerates the optimization process by seamlessly parallelizing the tuning across multiple nodes.
- APIs: REST and gRPC are supported for submitting optimization tasks.
- Code Less: When working with Hugging Face models, Neural Solution drives the optimization process without requiring any code modifications by integrating the Neural Coder's functionality.

# How does Nueral Solution Work?
![NS-OaaS-Intro](./docs/imgs/NS-OaaS-Intro.png)

# Get Started
## Installation
<details>
  <summary>Prerequisites</summary>

<!--TODO: Precise OS versions-->

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
- [APIs Reference](./docs/description_api.md)

# Contact

Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Solution related question.


