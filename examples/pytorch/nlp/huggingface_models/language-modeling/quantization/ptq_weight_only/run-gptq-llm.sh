python run-gptq-llm.py \
    --model_name_or_path /models/opt-125m \
    --dataset pile \
    --wbits 4 \
    --group_size 128 \
    --act-order \
    --gpu
