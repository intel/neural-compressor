1. eval fp32

```
python main.py --model_name_or_path <MODEL_NAME>
```


2. quantize

```
python main.py --model_name_or_path <MODEL_NAME> --int8
```

MODEL_NAME can be following (will use facebook/opt-125m if not specify --model_name_or_path):

facebook/opt-125m
facebook/opt-1.3b

3. sq

```
python main.py --model_name_or_path <MODEL_NAME> --int8 --sq
```

4. eval int8

TODO