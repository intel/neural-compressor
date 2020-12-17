Adaptor
=================
1. query fw capbility
2. parse tune config ( lpot config -> fwk capbility)
3. (optianal) pre optimize 
4. do the quantization



Extension
=================
Let us take onnxruntime as en example. Onnxruntime is a backend proposed by microsoft, and it's based on MLAS kernel defaultly. 
Onnxruntime already has  [quantization tools](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization), so the question becomes how to intergrate onnxruntime quantization tools into LPOT. 
1. capbility
   
   we should explore quantization capbility in first. According to [onnx_quantizer](https://github.com/microsoft/onnxruntime/blob/503b61d897074a494f5798069308ee67d8fb9ace/onnxruntime/python/tools/quantization/onnx_quantizer.py#L77), the quantization tools support following attributes:
   1.1 whether per_channel
   1.2 whether reduce_range
   1.3 QLinear mode or Integer mode (which is only seen in onnxruntime)
   1.4 whether static (static quantization or dynamci quantization)
   1.4 weight_qtype (choices are float32, int8 and uint8)
   1.5 input_qtype (choices are float32, int8 and uint8)
   1.6 quantization_params (None if dynamic quantization)
   1.7 &1.8 nodes_to_quantize, nodes_to_exclude
   1.9  op_types_to_quantize

   so we can pass a tune capbility to LPOT like

   ```yaml
   {'optypewise': {'conv': 
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']},
                    'algorithm': ['minmax', ],
                    'granularity': ['per_channel']
                   }, 
                   'matmul': 
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']},
                    'algorithm': ['minmax', ],
                    'granularity': ['per_channel']
                   }
                   }, 
    'opwise':  {('conv1', 'conv'):
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']}
                   }
                   }
    }
   ```

2. parse tune config
   
   LPOT will generate a tune config from your tune capbility like
   ```yaml
    {
        'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']],
        'fp32': [['CONV2D', 'RELU', 'BN']]}, 
        'calib_iteration': 10,
        'op': {
        ['op1', 'CONV2D']: {
            'activation':  {'dtype': 'uint8',
                            'algorithm': 'minmax',
                            'scheme':'sym',
                            'granularity': 'per_tensor'},
            'weight': {'dtype': 'int8',
                        'algorithm': 'kl',
                        'scheme':'asym',
                        'granularity': 'per_channel'}
        },
        ['op2', 'RELU]: {
            'activation': {'dtype': 'int8',
                            'scheme': 'asym',
                            'granularity': 'per_tensor',
                            'algorithm': 'minmax'}
        },
        ['op3', 'CONV2D']: {
            'activation':  {'dtype': 'fp32'},
            'weight': {'dtype': 'fp32'}
        },
        ...
        }
    }
   ```
   then you can parse this config into format that ONNXQuantizer can accept
   please make sure whether your quantization API support model wise or op wise quantization. for example, node "conv1" use "minmax" algorithm and node "conv2" use "KL" algorithm, or the whole model must use "minmax" or "KL" in general.

3. pre-optimize
   if your backend support FP32 graph optimization, you can apply it in **query_fw_capability** and quantize your optimized fp32 model instead of original model
   >model = self.pre_optimized_model if self.pre_optimized_model else model

4. do quantization
   
   This part depend on your backend implementationm you may refer to [onnxruntime](../lpot/adaptor/onnxrt.py) as an example.
