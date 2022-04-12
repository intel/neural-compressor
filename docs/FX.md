# FX

## Overview
FX is a PyTorch toolkit for developers to use to transform nn.Module instance. With converted torch.fx.GraphModule, we can automatically insert quant/dequant operation within PyTorch.  

 - `torch.fx.symbolic_trace()`

    Fake values, called Proxies, will be fed into model and record all operations in it. Then, we can get torch.fx.GraphModule from torch.nn.Module.  

-----------------------------------------

## Usage

**Note:** Make sure to replace backend with `pytorch_fx` in **conf.yaml**.  

Then, you can quantize model as usual with neural_compressor without editing the source code of the model, as shown below:   

```
if args.tune:
    from neural_compressor.experimental import Quantization, common
    model.eval()
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    q_model = quantizer.fit()
    q_model.save(args.tuned_checkpoint)
    return
```

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

    As symbolic trace cannot handle dynamic control, tensor iteration and so on, we might meet trace failure sometimes. In order to quantize the model successfully, we suggest two approaches to preprocess the model here.

    1. Non_traceable_module_class/name

        Select module classes or names that cannot be traced by proxy object, and pass them into neural_compressor: `common.Model` as a dict. 

        These non_traceable modules will be considered as a  called function. If there is any nn.Conv2D in modules, it won't be converted into quantized::Conv2D.

        **For example:** example/pytorch/fx/object_detection/maskrcnn/pytorch/tools/test_net.py

        ``` python
        prepare_custom_config_dict = {"non_traceable_module_class": [
            AnchorGenerator, RPNPostProcessor, Pooler, PostProcessor, \
            MaskRCNNFPNFeatureExtractor, MaskPostProcessor, FPN, RPNHead
            ]}
        quantizer.model = common.Model(model, \
            **{'prepare_custom_config_dict': prepare_custom_config_dict})
        ```

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

