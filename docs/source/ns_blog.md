# Neural Solution: Simplifying Model Quantization and Boosting Accuracy-aware Tuning with Intel® Neural Compressor

> Authors: Yi Liu, Kaihui Tang, Sihan Chen, Liang Lv, Feng Tian, and Haihao Shen, Intel Corporation

`Neural Solution`, `Intel Neural Compressor`, `Distributed Tuning`, `Deep Learning`, `Quantization`

In today's fast-paced world of deep learning, model compression techniques play a crucial role in enhancing efficiency and reducing computational resources. Intel® Neural Compressor (INC) is a cutting-edge tool that offers a wide range of popular model compression techniques, including quantization, pruning, distillation, and neural architecture search on mainstream frameworks. It supports a wide range of Intel hardware and has been extensively tested. The tool validates thousands of models from popular models by leveraging zero-code optimization solution [Neural Coder](https://github.com/intel/neural-compressor/blob/master/neural_coder#what-do-we-offer) and automatic [accuracy-driven](https://github.com/intel/neural-compressor/blob/master/docs/source/design.md#workflow) quantization strategies.

In this blog, we are happy to introduce Neural Solution, a novel component that brings the capabilities of INC as a service. Neural Solution simplifies the process of quantizing models and improves efficiency in accuracy-aware tuning.

### What is Neural Solution?

Neural Solution addresses the time-consuming process of model quantization by levering the distributed tuning and providing a convenient interface for seamless integration, allowing users to optimize their models without the need for repetitive environment setups or extensive code adaptation.

One major challenge in model quantization is identifying the optimal accuracy-relative configuration, including the selection of the appropriate algorithm for calibration and specific key operators to be fallbacks for lower precision. This configuration tuning process is time-consuming. To mitigate this pain point, Neural Solution allows users to parallelize the tuning process across multiple nodes by simply specifying the number of workers in the task request.

In addition, Neural solution also offers a convenient interface for seamless integration into different applications or platforms. It exposes both RESTful and gRPC APIs, empowering users to submit quantization tasks, query the optimization process, and obtain tuning results with ease.

Moreover, for the Hugging Face models, Neural Solution eliminates the need for any code modifications during the optimization process by seamlessly integrating the functionality of the Neural Coder. This approach significantly lowers the barrier to entry for users who may not possess extensive coding expertise.

[TBD,  how does it solve privacy concerns]


![NS-OaaS-Intro (1)](../../neural_solution/docs/source/imgs/NS-OaaS-Intro.png)

[The fig about how does the neural solution work]

[TBD some detail about how does the ns work?]

[TODO add links for keywords]

### Get started with neural solution

(By following the e2e example)

#### Install neural solution
#### start neural solution service
#### submit task
#### query result
#### stop service


### Conclusion/future work
In conclusion, Neural Solution is a powerful service that brings the optimization capabilities of Intel® Neural Compressor to users. It simplifies the processes of model quantization and accuracy-aware tuning, making them more accessible and efficient. We welcome your feedback and suggestions as we continue to enhance and improve Neural Solution.


Some notes(will be delete before release):

Outline
- Introduce INC with a few sentences
- Introduce NS with a few sentences
- The value of NS
  - efficient: distributed tuning
  - accessible: APIs
  - code less
- How to use it, following the doc of quantizing e2e hg model
- Conclusion


backup title:
1. Neural Solution: Bring the Optimization Power of Intel® Neural Compressor as a Service
