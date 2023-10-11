"This is a sample code for SignRound, which currently only supports LlaMa, OPT, and BLOOM models. We will provide a unified API that will support a broader range of models in Intel Neural Compressor"
# Prerequisite
python 3.9 or higher 
pip install -r requirements.txt


# Run
cd to current folder
CUDA_VISIBLE_DEVICES=0  python3 signround.py --model_name facebook/opt-125m --amp  --num_bits 4  --group_size -1 --seqlen 512

