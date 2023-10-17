Neural Coder as Python API
===========================

Neural Coder can be used as Python APIs. We currently provide 3 main user-facing APIs for Neural Coder: enable, bench and superbench.

#### Enable
Users can use ```enable()``` to enable specific features into DL scripts:
```
from neural_coder import enable
enable(
    code="neural_coder/examples/vision/resnet50.py",
    features=[
        "pytorch_jit_script",
        "pytorch_channels_last",
    ],
)
```
To run benchmark directly on the optimization together with the enabling:
```
from neural_coder import enable
enable(
    code="neural_coder/examples/vision/resnet50.py",
    features=[
        "pytorch_jit_script",
        "pytorch_channels_last"
    ],
    run_bench=True,
)
```

#### Bench
To run benchmark on your code with an existing patch:
```
from neural_coder import bench
bench(
    code="neural_coder/examples/vision/resnet50.py",
    patch_path="${your_patch_path}",
)
```

#### SuperBench
To sweep on optimization sets with a fixed benchmark configuration:
```
from neural_coder import superbench
superbench(code="neural_coder/examples/vision/resnet50.py")
```
To sweep on benchmark configurations for a fixed optimization set:
```
from neural_coder import superbench
superbench(
    code="neural_coder/examples/vision/resnet50.py",
    sweep_objective="bench_config",
    bench_feature=[
        "pytorch_jit_script",
        "pytorch_channels_last",
    ],
)
```
