# What's Neural Solution?

Neural Solution is a flexible and easy to use tool that brings the capabilities of Intel® Neural Compressor as a service. With Neural Solution, Users can effortlessly submit optimization tasks through the RESTful/gRPC APIs. Neural Solution automatically dispatches these tasks to one or multiple nodes, streamlining the entire process.

# Why Neural Solution?

- Task Parallelism: Neural Solution automatically schedules the optimization task queue by coordinating available resources and allows execution of multiple optimization tasks simultaneously.
- Tuning Parallelism: Neural Solution accelerates the optimization process by seamlessly parallelizing the tuning across multiple nodes.
- APIs Support: Neural Solution supports both RESTful and gRPC APIs, enabling users to conveniently submit optimization tasks.
- Code Less: When working with Hugging Face models, Neural Solution seamlessly integrates the functionality of the [Neural Coder](https://github.com/intel/neural-compressor/tree/master/neural_coder), eliminating the need for any code modifications during the optimization process.

# How does Neural Solution Work?
![NS-OaaS-Intro](./docs/source/imgs/NS-OaaS-Intro.png)

# Get Started
## Installation
### Prerequisites

- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/)
- Install [Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build)
- Python 3.8 or later


There are two ways to install the neural solution:
### Method 1. Using pip:
```
pip install neural-solution
```

### Method 2. Building from source:
```shell
# get source code
git clone https://github.com/intel/neural-compressor
cd neural-compressor

# install neural compressor
pip install -r requirements.txt
python setup.py install

# install neural solution
pip install -r neural_solution/requirements.txt
python setup.py neural_solution install
```

## End-to-end examples
- [Quantizing a Hugging Face model](./examples/hf_models/README.html)
- [Quantizing a custom model](./examples/custom_models_optimized/tf_example1/README.html)
## Learn More
<!-- TODO more docs(Install details, API and so on...) -->

- The Architecture documents
- [APIs Reference](./docs/source/description_api.html)

# Contact

Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Solution related question.
