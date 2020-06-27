# BERT

From [Gluonnlp Bert](https://github.com/dmlc/gluon-nlp/tree/v0.9.x/scripts/bert)


# Quantization with iLiT
## bert_base MRPC
```
 python3 finetune_classifier.py \
        --task_name MRPC \
        --only_inference \
        --model_parameters ./output_dir/model_bert_MRPC_4.params

```

## bert_base Squad
```
python3 finetune_squad.py \
        --model_parameters ./output_dir/net.params \
        --round_to 128 \
        --test_batch_size 128 \
        --only_predict
```
 

# Dependency

```
pip install mxnet-mkl gluonnlp

```
