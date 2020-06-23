# BERT

From [Gluonnlp Bert](https://github.com/dmlc/gluon-nlp/tree/v0.9.x/scripts/bert)


# Quantization with iLiT
```
python3 finetune_classifier.py \
        --task_name MRPC \
        --only_inference \
        --model_parameters ./output_dir/model_bert_MRPC_4.params \
        --auto_tuning

```
 

# Dependancy

```
pip install mxnet-mkl gluonnlp

```