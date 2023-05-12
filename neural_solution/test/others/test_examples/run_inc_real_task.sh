# source ~/miniconda3/etc/profile.d/conda.sh # replace it according to node
conda activate inc_kh # replace it with real enviroment name
# cd INC project workplace
cd /home/kaihui/code/multinode_tuning/neural-compressor
export PYTHONPATH=$PYTHONPATH:$PWD
# cd example path
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx
# run cmd
python -u ./run_glue.py --model_name_or_path distilbert_mrpc --task_name mrpc --do_eval  --max_seq_length 128 --per_device_eval_batch_size 16 --no_cuda --output_dir ./int8_model_dir --tune --overwrite_output_dir