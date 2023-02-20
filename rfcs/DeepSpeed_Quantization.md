## **Summary**

This is a proposal of extending the **quantization** capability of DeepSpeed.

DeepSpeed Compression has supported weight quantization and activation quantization.

In this proposal, we would like to enhance the quantization functionality by integrating post training static quantization supported by [Intel(R) Neural Compressor](https://github.com/intel/neural-compressor) into DeepSpeed.

## **Motivation**

Current state-of-the-art deep neural networks have a great number of parameters, which lead to their heavy computing resource consumption during training and inference. To address this issue, post training quantization has been proposed in order to make large models to be deployed on some edge devices with limited computation and memory footprint. 

It should be the first trail user to execute when applying low precision in their models as the post training static quantization doesn't request to do any training related works.

## **Proposed Implementation on Quantization**

As current implementation of DeepSpeed is focusing on compression during training

```json
   # new items added into compression config file
   "compression_inference": {         # new field
    "static_quantization":            # new field
        "enabled": true               # new field
    }
    
```

besides the changes in the compression config file, we also need introduce a new function `init_quantization()`. Different with `init_compression()`, it will only focus on post training static quantization.

```python

def init_quantization(model, calib_dataloader, deepspeed_config, mpu=None):
    ### this function is used to return a Quantizer object used in the following quantization.

class Quantizer():
    def __init__(self, model, calib_dataloader, deepspeed_config, mpu):
        ...
    
    def fit():
        ### the main entry of post training static quantization 

```

## **Quantization Results**
As for the structural sparsity results, please refer to below chart.

<a target="_blank" href="./quantization_result.png">
  <img src="./quantization_result.png" alt="Extension" width="80%" height="80%">
</a>

## **Summary**

This RFC is used to extend the capability of DeepSpeed compression and purse the best performance by running on frameworks 

