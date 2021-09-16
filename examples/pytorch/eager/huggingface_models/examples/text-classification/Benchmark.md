## original check:
eval_accuracy = 0.8382
throughput = 40.09

```
python run_glue.py  --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train  --do_eval  --max_seq_length 128  --per_device_train_batch_size 32   --learning_rate 2e-5   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir
```

## Lpot fine-tune:
Accuracy = 0.81341
throughput = 51.19

https://github.com/leonardozcm/neural-compressor/tree/master/examples/pytorch/eager/huggingface_models

```
export TASK_NAME=MRPC

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/

bash run_tuning.sh --topology=bert_base_MRPC --dataset_location=/root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad --input_model=/tmp/$TASK_NAME/


python run_glue_tune.py --tuned_checkpoint best_model --task_name MRPC --max_seq_length 128 --benchmark --int8 --output_dir /tmp/$TASK_NAME/ --model_name_or_path bert-base-cased
```


## Lpot fine-tune + prune:

pruning takes time > 15h

```
python examples/text-classification/run_glue_no_trainer_prune.py --task_name mnli --max_length 128 \
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured \
       --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3 --output_dir /tmp/$TASK_NAME/ \
       --prune --config prune.yaml --output_model prune_model/model.pt --seed 5143
```

## ONNX:
accuracy = 0.8603
throughput = 53.237

refer to https://github.com/intel/neural-compressor/tree/1e295885782c05f8a980d74a88c17311e03cf7aa/examples/onnxrt/language_translation/bert
```
bash prepare_data.sh --data_dir=./MRPC --task_name=$TASK_NAME
bash prepare_model.sh --input_dir=/root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad \
                      --task_name=$TASK_NAME \
                      --output_model=./bert.onnx # model path as *.onnx

python run_glue_tune.py  --task_name MRPC --max_seq_length 128  --output_dir /tmp/$TASK_NAME/ --model_name_or_path bert-base-cased
```

## bigdl-nano (jemalloc + omp):



