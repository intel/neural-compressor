export CUDA_VISIBLE_DEVICES=3
export http_proxy=http://child-jf.intel.com:912
export https_proxy=http://child-jf.intel.com:912

/home/cyy/anaconda3/envs/cyy_llava/bin/python examples/pytorch/multimodal/quantization/ptq/run_llava_no_trainer.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --image-folder /dataset/coco/images/train2017/ \
    --question-file /data4/cyy/gptq_inc/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
    
