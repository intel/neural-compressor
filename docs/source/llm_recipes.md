## LLMs Quantization Recipes

Intel® Neural Compressor supported advanced large language models (LLMs) quantization technologies including SmoothQuant (SQ) and Weight-Only Quant (WOQ),
and verified a list of LLMs on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with [PyTorch](https://pytorch.org/),
[Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) and [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers).  
This document aims to publish the specific recipes we achieved for the popular LLMs and help users to quickly get an optimized LLM with limited 1% accuracy loss.

> Notes:
>
> - The quantization algorithms provide by [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and the evaluate functions provide by [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers).
> - The model list are continuing update, please expect to find more LLMs in the future.

## Large Language Models Recipes

|             Models              | SQ INT8 | WOQ INT8 | WOQ INT4 |
| :-----------------------------: | :-----: | :------: | :------: |
|       EleutherAI/gpt-j-6b       |    ✔    |    ✔     |    ✔     |
|        facebook/opt-1.3b        |    ✔    |    ✔     |    ✔     |
|        facebook/opt-30b         |    ✔    |    ✔     |    ✔     |
|    meta-llama/Llama-2-7b-hf     |   WIP   |    ✔     |    ✔     |
|    meta-llama/Llama-2-13b-hf    |   WIP   |    ✔     |    ✔     |
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
|        THUDM/chatglm2-6b        |   WIP   |    ✔     |   WIP    |
|        THUDM/chatglm3-6b        |   WIP   |    ✔     |    ✔     |

**Detail recipes can be found [HERE](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/llm_quantization_recipes.md).**

> Notes:
>
> - This model list comes from [IPEX](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/llm.html).
> - The WIP recipes will be published soon.

## Large Language Models Accuracy

<table><thead>
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
  </tr></thead>
<tbody>
  <tr>
    <td>baichuan-inc/Baichuan-13B-Chat</td>
    <td>67.57%</td>
    <td>67.86%</td>
    <td>1.0043</td>
    <td>67.55%</td>
    <td>0.9997</td>
    <td>67.46%</td>
    <td>0.9984</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>baichuan-inc/Baichuan2-13B-Chat</td>
    <td>71.51%</td>
    <td>75.51%</td>
    <td>1.0559</td>
    <td>71.57%</td>
    <td>1.0008</td>
    <td>71.45%</td>
    <td>0.9992</td>
    <td>70.87%</td>
    <td>0.9911</td>
  </tr>
  <tr>
    <td>baichuan-inc/Baichuan2-7B-Chat</td>
    <td>67.67%</td>
    <td>67.51%</td>
    <td>0.9976</td>
    <td>67.61%</td>
    <td>0.9991</td>
    <td>68.08%</td>
    <td>1.0061</td>
    <td>67.18%</td>
    <td>0.9928</td>
  </tr>
  <tr>
    <td>bigscience/bloom-1b7</td>
    <td>46.34%</td>
    <td>47.97%</td>
    <td>1.0352</td>
    <td>46.21%</td>
    <td>0.9972</td>
    <td>47.00%</td>
    <td>1.0142</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>databricks/dolly-v2-12b</td>
    <td>64.35%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>63.92%</td>
    <td>0.9933</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6b</td>
    <td>68.31%</td>
    <td>68.00%</td>
    <td>0.9955</td>
    <td>68.27%</td>
    <td>0.9994</td>
    <td>68.23%</td>
    <td>0.9988</td>
    <td>67.40%</td>
    <td>0.9867</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-neox-20b</td>
    <td>72.33%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>72.29%</td>
    <td>0.9994</td>
    <td>72.15%</td>
    <td>0.9975</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>facebook/opt-1.3b</td>
    <td>57.89%</td>
    <td>57.35%</td>
    <td>0.9907</td>
    <td>58.12%</td>
    <td>1.0040</td>
    <td>58.01%</td>
    <td>1.0021</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>facebook/opt-30b</td>
    <td>71.49%</td>
    <td>71.51%</td>
    <td>1.0003</td>
    <td>71.53%</td>
    <td>1.0006</td>
    <td>71.82%</td>
    <td>1.0046</td>
    <td>71.43%</td>
    <td>0.9992</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-13b-hf</td>
    <td>76.77%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>76.89%</td>
    <td>1.0016</td>
    <td>76.96%</td>
    <td>1.0025</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-70b-hf</td>
    <td>79.64%</td>
    <td>79.53%</td>
    <td>0.9986</td>
    <td>79.62%</td>
    <td>0.9997</td>
    <td>80.05%</td>
    <td>1.0051</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-7b-hf</td>
    <td>73.92%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>73.90%</td>
    <td>0.9997</td>
    <td>73.51%</td>
    <td>0.9945</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>mistralai/Mistral-7B-v0.1</td>
    <td>75.90%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>75.80%</td>
    <td>0.9987</td>
    <td>75.37%</td>
    <td>0.9930</td>
    <td>75.82%</td>
    <td>0.9989</td>
  </tr>
  <tr>
    <td>THUDM/chatglm2-6b</td>
    <td>53.23%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>53.00%</td>
    <td>0.9957</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>THUDM/chatglm3-6b</td>
    <td>59.09%</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>59.03%</td>
    <td>0.9990</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>58.59%</td>
    <td>0.9915</td>
  </tr>
  <tr>
    <td>tiiuae/falcon-40b</td>
    <td>77.22%</td>
    <td>77.26%</td>
    <td>1.0005</td>
    <td>77.18%</td>
    <td>0.9995</td>
    <td>77.97%</td>
    <td>1.0097</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>tiiuae/falcon-7b</td>
    <td>74.67%</td>
    <td>76.17%</td>
    <td>1.0201</td>
    <td>74.73%</td>
    <td>1.0008</td>
    <td>74.79%</td>
    <td>1.0016</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
</tbody></table>
