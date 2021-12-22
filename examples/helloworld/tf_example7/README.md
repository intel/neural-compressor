tf_example7 example
=====================
This example is used to demonstrate how to utilize Neural Compressor to enabling quantization and benchmark with python-flavor config.

1. Prepare Model
```shell
pip install intel-tensorflow==2.7.0
python train.py
```

2. Use python code to set necessary parameters
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

3. Run quantization and benchmark
```python
    from neural_compressor.experimental import Quantization, Benchmark, common
    evaluator = Benchmark(conf)
    evaluator.model = common.Model("../models/frozen_graph.pb")
    evaluator('performance')

    quantizer = Quantization(conf)
    quantizer.model = common.Model("../models/frozen_graph.pb")
    quantizer()
```

```shell
    python test.py
``` 

