pip install -r requirements.txt



CUDA_VISIBLE_DEVICES=0  python3 signround.py --model_name facebook/opt-125m --amp   --num_bits 4  --group_size -1

