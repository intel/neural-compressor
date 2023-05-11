1. Get fp32 accuracy

```
python main.py --model_name_or_path <MODEL_NAME>
```


2. Do int8 quantization

```
python main.py --model_name_or_path <MODEL_NAME> --int8
```

`<MODEL_NAME>` can be following:

- gpt2
- gpt2-medium
- facebook/opt-125m

3. Do int8 smooth quantization

```
python main.py --model_name_or_path <MODEL_NAME> --int8 --sq
```