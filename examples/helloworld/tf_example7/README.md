tf_example7 example
=====================
This example is used to demonstrate how to utilize Neural Compressor to enabling quantization and benchmark with python-flavor config.

### 1. Installation
```shell
pip install -r requirements.txt
```

### 2. Prepare Model
```shell
python train.py
```

### 3. Run Command
* Run quantization
```shell
python test.py --tune
``` 
* Run benchmark
```shell
python test.py --benchmark
``` 

### 4. Introduction  
* Use python code to set necessary parameters.
```python
    from neural_compressor import conf
    dataloader = {
        'dataset': {'dummy_v2': {'input_shape': [28, 28]}}
    }
    conf.evaluation.performance.dataloader = dataloader
    conf.quantization.calibration.dataloader = dataloader
    conf.evaluation.accuracy.dataloader = dataloader
    conf.tuning.accuracy_criterion.absolute = 0.9
```
* Run quantization and benchmark. 
```python
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization(conf)
    quantizer.model = common.Model("../models/frozen_graph.pb")
    quantizer.fit()
    
    from neural_compressor.experimental import Benchmark
    evaluator = Benchmark(conf)
    evaluator.model = common.Model("../models/frozen_graph.pb")
    evaluator('performance')
    
```

