## LLMs Quantization Recipes

Intel® Neural Compressor supported advanced large language models (LLMs) quantization technologies including SmoothQuant (SQ) and Weight-Only Quant (WOQ),
and verified a list of LLMs on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with [PyTorch](https://pytorch.org/),
[Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) and [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers).  
This document aims to publish the specific recipes we achieved for the popular LLMs and help users to quickly get an optimized LLM with limited 1% accuracy loss.

> Notes:
>
> - The quantization algorithms provide by [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and the evaluate functions provide by [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers).
> - The model list are continuing update, please expect to find more LLMs in the future.

## IPEX key models recipes

|             Models              | SQ INT8 | WOQ INT8 | WOQ INT4 |
| :-----------------------------: | :-----: | :------: | :------: |
|       EleutherAI/gpt-j-6b       |    ✔    |    ✔     |    ✔     |
|        facebook/opt-1.3b        |    ✔    |    ✔     |    ✔     |
|        facebook/opt-30b         |    ✔    |    ✔     |    ✔     |
|    meta-llama/Llama-2-7b-hf     |    ✔    |    ✔     |    ✔     |
|    meta-llama/Llama-2-13b-hf    |    ✔    |    ✔     |    ✔     |
|    meta-llama/Llama-2-70b-hf    |    ✔    |    ✔     |    ✔     |
|        tiiuae/falcon-7b         |    ✔    |    ✔     |    ✔     |
|        tiiuae/falcon-40b        |    ✔    |    ✔     |    ✔     |
| baichuan-inc/Baichuan-13B-Chat  |    ✔    |    ✔     |    ✔     |
| baichuan-inc/Baichuan2-13B-Chat |    ✔    |    ✔     |    ✔     |
| baichuan-inc/Baichuan2-7B-Chat  |    ✔    |    ✔     |    ✔     |
|      bigscience/bloom-1b7       |    ✔    |    ✔     |    ✔     |
|     databricks/dolly-v2-12b     |    ✖    |    ✔     |    ✖     |
|     EleutherAI/gpt-neox-20b     |    ✖    |    ✔     |    ✔     |
|    mistralai/Mistral-7B-v0.1    |    ✖    |    ✔     |    ✔     |
|        THUDM/chatglm2-6b        |   WIP   |    ✔     |    ✔     |
|        THUDM/chatglm3-6b        |   WIP   |    ✔     |    ✔     |

**Detail recipes can be found [HERE](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/llm_quantization_recipes.md).**

> Notes:
>
> - This model list comes from [IPEX](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/llm.html).
> - The WIP recipes will be published soon.

## IPEX key models accuracy
<table>
<thead>
  <tr>
    <th rowspan="3">Model</th>
    <th colspan="9">lambada_openai</th>
  </tr>
  <tr>
    <th>FP32</th>
    <th colspan="2">SQ INT8</th>
    <th colspan="2">WOQ INT8</th>
    <th colspan="2">WOQ INT4 GPTQ</th>
    <th colspan="2">WOQ INT4 AutoRound</th>
  </tr>
  <tr>
    <th>ACC</th>
    <th>ACC</th>
    <th>Ratio</th>
    <th>ACC</th>
    <th>Ratio</th>
    <th>ACC</th>
    <th>Ratio</th>
    <th>ACC</th>
    <th>Ratio</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>baichuan-inc/Baichuan-13B-Chat</td>
    <td>67.57%</td>
    <td>68.23%</td>
    <td>1.0098</td>
    <td>67.57%</td>
    <td>1.0000</td>
    <td>67.84%</td>
    <td>1.0040</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>baichuan-inc/Baichuan2-13B-Chat</td>
    <td>71.51%</td>
    <td>70.89%</td>
    <td>0.9913</td>
    <td>71.53%</td>
    <td>1.0003</td>
    <td>71.76%</td>
    <td>1.0035</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>baichuan-inc/Baichuan2-7B-Chat</td>
    <td>67.67%</td>
    <td>67.96%</td>
    <td>1.0043</td>
    <td>67.59%</td>
    <td>0.9988</td>
    <td>67.24%</td>
    <td>0.9936</td>
    <td>67.42%</td>
    <td>0.9963</td>
  </tr>
  <tr>
    <td>bigscience/bloom-1b7</td>
    <td>46.34%</td>
    <td>47.99%</td>
    <td>1.0356</td>
    <td>46.38%</td>
    <td>1.0009</td>
    <td>46.19%</td>
    <td>0.9968</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>databricks/dolly-v2-12b</td>
    <td>64.35%</td>
    <td>NA</td>
    <td>NA</td>
    <td>64.10%</td>
    <td>0.9961</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6b</td>
    <td>68.31%</td>
    <td>68.33%</td>
    <td>1.0003</td>
    <td>68.23%</td>
    <td>0.9988</td>
    <td>68.79%</td>
    <td>1.0070</td>
    <td>68.43%</td>
    <td>1.0018</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-neox-20b</td>
    <td>72.33%</td>
    <td>NA</td>
    <td>NA</td>
    <td>72.25%</td>
    <td>0.9989</td>
    <td>71.96%</td>
    <td>0.9949</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>facebook/opt-1.3b</td>
    <td>57.89%</td>
    <td>57.54%</td>
    <td>0.9940</td>
    <td>58.08%</td>
    <td>1.0033</td>
    <td>58.57%</td>
    <td>1.0117</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>facebook/opt-30b</td>
    <td>71.49%</td>
    <td>71.51%</td>
    <td>1.0003</td>
    <td>71.51%</td>
    <td>1.0003</td>
    <td>71.82%</td>
    <td>1.0046</td>
    <td>72.11%</td>
    <td>1.0087</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-13b-hf</td>
    <td>76.77%</td>
    <td>76.25%</td>
    <td>0.9932</td>
    <td>76.75%</td>
    <td>0.9997</td>
    <td>77.43%</td>
    <td>1.0086</td>
    <td>76.75%</td>
    <td>0.9997</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-70b-hf</td>
    <td>79.64%</td>
    <td>79.55%</td>
    <td>0.9989</td>
    <td>79.57%</td>
    <td>0.9991</td>
    <td>80.09%</td>
    <td>1.0057</td>
    <td>79.97%</td>
    <td>1.0041</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-7b-hf</td>
    <td>73.92%</td>
    <td>73.45%</td>
    <td>0.9936</td>
    <td>73.96%</td>
    <td>1.0005</td>
    <td>73.45%</td>
    <td>0.9936</td>
    <td>73.49%</td>
    <td>0.9942</td>
  </tr>
  <tr>
    <td>mistralai/Mistral-7B-v0.1</td>
    <td>75.90%</td>
    <td>NA</td>
    <td>NA</td>
    <td>75.80%</td>
    <td>0.9987</td>
    <td>76.13%</td>
    <td>1.0030</td>
    <td>75.61%</td>
    <td>0.9962</td>
  </tr>
  <tr>
    <td>THUDM/chatglm2-6b</td>
    <td>53.23%</td>
    <td>NA</td>
    <td>NA</td>
    <td>53.19%</td>
    <td>0.9992</td>
    <td>52.77%</td>
    <td>0.9914</td>
    <td>53.35%</td>
    <td>1.0023</td>
  </tr>
  <tr>
    <td>THUDM/chatglm3-6b</td>
    <td>59.09%</td>
    <td>NA</td>
    <td>NA</td>
    <td>59.01%</td>
    <td>0.9986</td>
    <td>NA</td>
    <td>NA</td>
    <td>58.61%</td>
    <td>0.9919</td>
  </tr>
  <tr>
    <td>tiiuae/falcon-40b</td>
    <td>77.22%</td>
    <td>77.04%</td>
    <td>0.9977</td>
    <td>77.22%</td>
    <td>1.0000</td>
    <td>77.94%</td>
    <td>1.0093</td>
    <td>78.79%</td>
    <td>1.0203</td>
  </tr>
  <tr>
    <td>tiiuae/falcon-7b</td>
    <td>74.67%</td>
    <td>76.44%</td>
    <td>1.0237</td>
    <td>74.77%</td>
    <td>1.0013</td>
    <td>75.00%</td>
    <td>1.0044</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
</tbody>
</table>