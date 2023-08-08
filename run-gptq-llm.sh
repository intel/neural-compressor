export CUDA_VISIBLE_DEVICES=2
python examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only/run-gptq-llm.py \
    --model_name_or_path /models/Llama-2-7b-hf/ \
    --dataset pile \
    --wbits 4 \
    --group_size 128 \
    --act-order \
    --gpu
