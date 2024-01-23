This is a sample code for SignRound ([arxiv](https://arxiv.org/abs/2309.05516)), which currently only supports LlaMa, OPT, and BLOOM models. We will provide a unified API that will support a broader range of models in Intel Neural Compressor.

![overview](./overview.png)


# Prerequisite
-python 3.9 or higher

- The transformers version required varies across different types of models. Here, the transformers version used for running models during experiments is provided as a reference.
    | Model | Transformers version |
    |  :----: | :----: |
    | decapoda-research/llama-7b-hf | 4.28 |
    | huggyllama/llama-7b | 4.28/4.30/4.34/4.36 |
    | meta-llama/Llama-2-7b-hf | 4.28/4.30/4.34/4.36 |
    | facebook/opt-6.7b | 4.28/4.30/4.34/4.36 |
    | bigscience/bloom-7b1 | 4.28/4.30/4.34/4.36 |

Please note that all experimental data in the paper is based on transformer version 3.28.1. the huggingface source for llama-7b-hf mentioned in the paper, 'decapoda-research/llama-7b-hf', is currently unavailable. You may opt for 'huggyllama/llama-7b' as an alternative, but please be aware that this replacement might yield slight differences in results. 



# Run

```bash
CUDA_VISIBLE_DEVICES=0  python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --seqlen 512
```

To optimize GPU memory usage, you can enable the "low_gpu_mem_usage" option. Additionally, you can reduce the training batch size (train_bs) and increase the gradient_accumulate_steps accordingly.

```bash
CUDA_VISIBLE_DEVICES=0 python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --seqlen 512 --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
## Known issue
To address the original lambada evaluation bug in the old version of lm-eval, we have incorporated the lm-eval from intel extension for transformers(ITREX). This discrepancy may lead to certain variations.

To reproduce our results in the paper, please install ITREX

```bash
pip install intel-extension-for-transformers
```
## Reference
If you find SignRound useful or relevant to your research, please kindly cite our paper

```
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```



