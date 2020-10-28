Intorduction
=============
If you don't care accuracy and want to measure performance only, you can follow examples/tensorflow/oob_models/README.md to quantize a frozen pb with dummy data and without accuracy constrains, after the tuning you can also run the quantized model to get performance information.

The oob_models already supported a lot of public topologies, please see a list of models defined in the "models_need_name" variable of run_tunin.mples/tensorflow/oob_models/run_tuning.sh. If you want to tune a new model, for example the examples/helloworld/frozen_models/simple_frozen_graph.pb without accuracy constrains, please follow the steps:


1. Add the model name in examples/tensorflow/oob_models/run_tuning.sh "models_need_name" list:
```
 models_need_name=(
+hello_world
 efficientnet-b0
 efficientnet-b0_auto_aug
 efficientnet-b5

```

2. Add the model input and output tensor information to examples/tensorflow/oob_models/model_detail.py 
```
 models = [
+    {
+        'model_name': 'hello_world',
+        'input': {'x': generate_data([28, 28])},
+        'output': ['Identity']
+    },
+
```
3. Run quantization:
```
 cd <Installation Folder>/examples/tensorflow/oob_models 
 ./run_tuning.sh --topology=hello_world --dataset_location= --input_model=<Installation Folder>/examples/helloworld/frozen_models/simple_frozen_graph.pb --output_model=<Installation Folder>/examples/helloworld/tf1.x_dummy/simple_frozen_graph_int8.pb
```
4. Run performance with quantized frozen pb
You can get the latency and throughput information:
```
 cd <Installation Folder>/examples/tensorflow/oob_models 
 ./run_benchmark.sh --topology=hello_world --dataset_location= --input_model=<Installation Folder>/examples/helloworld/tf1.x_dummy/simple_frozen_graph_int8.pb --mode=benchmark --batch_size=1 --iters=200
......
Batch size = 1
Latency: xxx ms
Throughput: xxx images/sec
```


