
# AutoRound: Advanced Weight-Only Quantization Algorithm for a Broad Range of Models

AutoRound is an advanced weight-only quantization algorithm, based on SignRound. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound. However, it comes at the cost of approximately 2.5 times the quantization runtime.

## Prerequisites
- Python 3.9 or higher

## Installation
Install the necessary dependencies with the following command:
```bash
pip install -r requirements.txt
```

## Usage
- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --enable_minmax_tuning --use_quant_input
```
- **Reduced GPU Memory Usage and Adjusted Training Batch Size:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
- **Utilizing the AdamW Optimizer:**
Include the flag `--adam`. Note that AdamW may be slightly less effective than Sign gradient descent in certain scenarios.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 signround.py --model_name facebook/opt-125m --amp --num_bits 4 --group_size -1 --iters 400 --lr 0.0025 --minmax_lr 0.0025 
```
It's recommended to use `--enable_minmax_tuning`.

## Tips
Consider increasing tuning steps and adjusting the learning rate based on a scaling law to achieve better results, albeit with increased running time. For instance, at step 800, a learning rate of 0.00125 could be employed.

## Known Issues
Auto Rounding may encounter random issues with Qwen models.

## Reference
If you find SignRound useful for your research, please cite our paper:
```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```