AutoRound provides an advanced weight-only quantization algorithm for broad models based on SignRound. It can achieve noticeable improvements, often significantly improvements than those of SignRound, at the cost of approximately 2.5 times the quantization runtime."

# Prerequisite
python 3.9 or higher

pip install -r requirements.txt


# Run

```bash
CUDA_VISIBLE_DEVICES=0  python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --enalbe_minmax_tuning --use_quant_input
```

To optimize GPU memory usage, you can enable the "low_gpu_mem_usage" option. Additionally, you can reduce the training batch size (train_bs) and increase the gradient_accumulate_steps accordingly.

```bash
CUDA_VISIBLE_DEVICES=0 python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1  --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
To utilize the AdamW optimizer, simply add the flag --adam. AdamW is typically slightly less effective than Sign gradient descent in our tested scenario.

To run original SignRound, please refer to the following cmd, --enalbe_minmax_tuning is recommended

```bash
CUDA_VISIBLE_DEVICES=0  python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1  --iters 400 --lr 0.0025 --minmax_lr 0.0025 
```

## Known issue
Qwen models have random issue in Auto Rounding

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

