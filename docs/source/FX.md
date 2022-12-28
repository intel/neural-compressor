FX
====
1. [Introduction](#introduction)
2. [FX Mode Support Matrix in Neural Compressor](#fx-mode-support-matrix-in-neural-compressor)
3. [Get Start](#get-start)

    3.1. [Post Training Static Quantization](#post-training-static-quantization)

    3.2. [Post Training Dynamic Quantization](#post-training-dynamic-quantization)

    3.3. [Quantization-Aware Training](#quantization-aware-training)

4. [Examples](#examples)
5. [Note](#note)
6. [Common Problem](#common-problem)

## Introduction
FX is a PyTorch toolkit for developers to use to transform nn.Module instance. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation.

With converted torch.fx.GraphModule, we can resolve three problems in quantization:
1. Automatically insert quant/dequant operation within PyTorch.
2. Use FloatFunctional to wrap tensor operations that require special handling for quantization into modules. Examples are operations like add and cat which require special handling to determine output quantization parameters.
3. Fuse modules: combine operations/modules into a single module to obtain higher accuracy and performance. This is done using the fuse_modules() API, which takes in lists of modules to be fused. We currently support the following fusions: [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu].

For detailed description, please refer to [PyTorch FX](https://pytorch.org/docs/stable/fx.html) and [FX Graph Mode Quantization](https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization)


## FX Mode Support Matrix in Neural Compressor

|quantization           |FX           |
|-----------------------|:-----------:|
|Static Quantization    |&#10004;     |
|Dynamic Quantization   |&#10004;     |
|Quantization-Aware Training         |&#10004;     |


## Get Start

**Note:** "backend" field indicates the backend used by the user in configure. And the "default" value means it will quantization model with fx backend for PyTorch model.

### Post Training Static Quantization

```
    from neural_compressor import quantization, PostTrainingQuantConfig
    conf = PostTrainingQuantConfig(backend="default")
    model.eval()
    q_model = quantization.fit(model, conf, calib_dataloader=dataloader, eval_func=eval_func)
    q_model.save("save/to/path")
```

### Post Training Dynamic Quantization

```
    from neural_compressor import quantization, PostTrainingQuantConfig
    conf = PostTrainingQuantConfig(backend="default", approach="dynamic")
    model.eval()
    q_model = quantization.fit(model, conf, eval_func=eval_func)
    q_model.save("save/to/path")
```

### Quantization-Aware Training

```
    from neural_compressor import QuantizationAwareTrainingConfig
    from neural_compressor.training import prepare_compression
    conf = QuantizationAwareTrainingConfig(backend="default")
    compression_manager = prepare_compression(model, conf)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model
    model.train()
    ####### Training loop #####

    compression_manager.save("save/to/path")
```

## Examples

User could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/fx/README.md) on how to quantize a model with FX backend.

## Note
Right now, we support auto quantization method and can avoid below common problem.   
For users, you will see log output below if you model failed on symbolic trace method.
```
[INFO] Fx trace of the entire model failed. We will conduct auto quantization
```

### Details
 - We combine GraphModule from symbolic_trace and imperative control flow. Therefore, the INT8 model consists of lots of GraphModules.

-----------------------------------------

## Common Problem

### *Dynamic Quantization*  

 - PyTorch Version: 1.9 or higher  

    You can use pytorch backend for dynamic quantization, there is no difference between pytorch and pytorch_fx. we don't need to trace model because we don't need to modify the source code of the model.  

### *Static Quantization* & *Quantization Aware Training*  

 - PyTorch Version: 1.8 or higher

    As symbolic trace cannot handle dynamic control, tensor iteration and so on, we might meet trace failure sometimes. In order to quantize the model successfully, we suggest user to trace model with below two approaches first and then pass it to neural_compressor.

    1. Non_traceable_module_class/name

        Select module classes or names that cannot be traced by proxy object, and pass them into prepare_fx as a dict. 

        **Please refer to:** https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html?highlight=non_traceable_module_class


    2. Decorator: @torch.fx.wrap

        If untraceable part is not a part of a module, like a global function called, or you want to move untraceable part out of model to keep other parts get quantized, you should try to use Decorator `@torch.fx.wrap`. The wrapped function must be in the global, not in the class.

        **For example:** examples/pytorch/fx/object_detection/ssd_resnet34/ptq/python/models/ssd_r34.py

        ``` python
        @torch.fx.wrap
        def bboxes_labels_scores(bboxes, probs, criteria = 0.45, max_output=200):
            boxes = []; labels=[]; scores=[]
            for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
                bbox = bbox.squeeze(0)
                prob = prob.squeeze(0)
                dbox,dlabel,dscore=decode_single(bbox, prob, criteria, max_output)
                boxes.append(dbox)
                labels.append(dlabel)
                scores.append(dscore)
            return [boxes,labels,scores]
            ```

