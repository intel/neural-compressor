TODO

- [x] sth about service, what is service why ,oaas
- [x] one of the key value of task level parallelism
- [ ] enhance the diagram
- [x] add future work
- [ ] clean the blog: remove the comment, something like `[xxx]`

backup title list:

- Neural Solution: Streamlining Model Quantization and Enhancing Efficiency with Intel® Neural Compressor as a Service
- Neural Solution: Simplifying Model Quantization and Efficiency with Intel® Neural Compressor as a Service
- **Neural Solution: Bring the Optimization Power of Intel® Neural Compressor as a Service**



# Neural Solution: Streamlining Model Optimization as a Service with Parallelism and Seamless Integration

> Authors: ...

Keywords: Neural Solution, Intel Neural Compressor, Distributed Tuning, Deep Learning, Quantization

In today's fast-paced world of deep learning, model compression techniques play a crucial role in enhancing efficiency and reducing computational resources. [Intel® Neural Compressor](https://github.com/intel/neural-compressor) (INC) is a cutting-edge tool that offers a wide range of popular model compression techniques, including quantization, pruning, distillation, and neural architecture search on mainstream frameworks. It supports a wide range of Intel hardware and has been extensively tested. The tool validates thousands of models by leveraging zero-code optimization solution Neural Coder and automatic accuracy-driven quantization strategies. [The paint point]However, the absence of service porting adds an additional burden to seamlessly integrate INC into existing systems or workflows. Furthermore, the lack of parallel task handling restricts scalability and hampers the efficient processing of multiple optimization requests. In this blog, we're happy to introduce [Neural Solution](https://github.com/intel/neural-compressor/tree/master/neural_solution), an new component that bring the optimization capabilities of INC as a service. Neural Solution simplifies the model quantization process and enhances efficiency in accuracy-aware tuning.



## What is Neural Solution?

Neural Solution provide task-level and tuning-level parallelism, coordinating the optimization task queue and leveraging distributed tuning to speedup optimization process. It also offers a seamless integration interface, eliminating the need for repetitive environment setups and code adaptation, simplifying the optimization process for users.

Neural Solution efficiently schedules the optimization task queue by coordinating available resources and tracking the execution status of each task. This concurrent scheduling ensures optimal resource utilization and allows for efficient execution of multiple optimization tasks simultaneously.

One major challenge in model quantization is identifying the optimal accuracy-relative configuration which is time-consuming. To mitigate this pain point, Neural Solution allows users to parallelize the tuning process across multiple nodes by simply specifying the number of workers in the task request.

In addition, Neural solution also offers a convenient interface for seamless integration into different applications or platforms. It exposes both RESTful and gRPC APIs, empowering users to submit quantization tasks, query the optimization process, and obtain tuning results with ease.

Moreover, for the Hugging Face models, Neural Solution eliminates the need for any code modifications during the optimization process by seamlessly integrating the functionality of the Neural Coder. This approach significantly lowers the barrier to entry for users who may not possess extensive coding expertise.


![NS-OaaS-Intro (1)](../../neural_solution/docs/source/imgs/NS-OaaS-Intro.png)

Fig 1. how does the neural solution work



## Get started with neural solution

Let's get start Neural Solution with an end-to-end example that quantizes a [Text classification model](https://github.com/huggingface/transformers/tree/v4.21-release/examples/pytorch/text-classification) from Hugging Face.

### Install neural solution

```shell
# get source code
git clone https://github.com/intel/neural-compressor
cd neural-compressor

# install neural solution
pip install -r neural_solution/requirements.txt
python setup.py neural_solution install
```

> More installation options and details can be found [here](https://github.com/intel/neural-compressor/tree/master/neural_solution#installation).

### Start the Neural Solution Service

```shell
# Start neural solution service with default configuration, log will be saved in the "serve_log" folder.
neural_solution start

# Start neural solution service with custom configuration
neural_solution start --task_monitor_port=22222 --result_monitor_port=33333 --restful_api_port=8001

# Help Manual
neural_solution -h
```


### Submit optimization task

- Step 1: Prepare the json file includes request content.

```shell
[user@server hf_model]$ cd path/to/neural_solution/examples/hf_model
[user@server hf_model]$ cat task_request.json
{
    "script_url": "https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py",
    "optimized": "False",
    "arguments": [
        "--model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result"
    ],
    "approach": "static",
    "requirements": [],
    "workers": 1
}
```


- Step 2: Submit the task request to service, and it will return the submit status and task id for future use.

```shell
[user@server hf_model]$ curl -H "Content-Type: application/json" --data @./task.json  http://localhost:8000/task/submit/

# response if submit successfully
{
    "status": "successfully",
    "task_id": "cdf419910f9b4d2a8320d0e420ac1d0a",
    "msg": "Task submitted successfully"
}
```

### Query optimization result

- Query the task status and result according to the `task_id`.

``` shell
[user@server hf_model]$ curl  -X GET  http://localhost:8000/task/status/{task_id}

# return the task status
{
    "status": "done",
    "optimized_result": {
        "optimization time (seconds)": "58.15",
        "accuracy": "0.3162",
        "duration (seconds)": "4.6488"
    },
    "result_path": "/path/to/projects/neural solution service/workspace/fafdcd3b22004a36bc60e92ec1d646d0/q_model_path"
}

```

### Stop the service

```shell
neural_solution stop
```

## Conclusion & Future work

Neural Solution provides users with the optimization capabilities of Intel® Neural Compressor as a service, simplifying model quantization and accuracy-aware tuning. In the future, we plan to port more optimization capabilities of INC, such as pruning and orchestration, and enhance deployment ability through Docker technology for greater adaptability. Your valuable feedback is appreciated as we strive to improve Neural Solution.