export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false

python eval_official.py \
    --model hf-causal-experimental \
    --model_args "pretrained=/models/opt-125m/,tokenizer=/models/opt-125m/,dtype=float16,trust_remote_code=True,add_special_tokens=False" \
    --tasks copa,wikitext2 \
    --device "cuda:0"
