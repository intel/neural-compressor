export CUDA_VISIBLE_DEVICES=6
python examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only/run-gptq-llm.py \
    --model_name_or_path /models/opt-125m/ \
    --weight_only_algo GPTQ \
    --dataset pile \
    --wbits 4 \
    --group_size 128 \
    --act-order \
    --gpu
