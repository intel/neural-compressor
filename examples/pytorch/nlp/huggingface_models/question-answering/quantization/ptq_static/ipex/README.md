# Bert_Large Inference

## Prepare

Follow [link](https://github.com/intel-innersource/frameworks.ai.models.intel-models/blob/develop/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

- Install dependency
```
  conda install intel-openmp
```

- Set ENV to use AMX if you are using SPR
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

- Install Intel® Extension for PyTorch* (IPEX)

```
  python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

> Note: Intel® Extension for PyTorch* has PyTorch version requirement. Please check more detailed information via the URL below.

- Install transformers and set tag to v3.0.2
```
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v3.0.2
  pip install -e ./
  cd ../
  ```

- Download dataset
  Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get dev-v1.1.json

- Downliad fine-tuned model
```
  mkdir bert_squad_model
  wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
  wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
```


## Run
```
  python run_qa.py 
    --model_type bert 
    --model_name_or_path ./bert_squad_model/ #finetuned model
    --do_lower_case 
    --predict_file ./dev-v1.1.json #dataset
    --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad 
    --do_eval 
    --max_seq_length 384 
    --doc_stride 128  
    --no_cuda  
    --tune 
    --output_dir ./savedresult  
    --int8 
    --int8_fp32
```

## Quick start
```
  bash run_tuning.sh --dataset_location=/path/to/dataset --input_model=/path/to/model 
```
```
  bash run_benchmark.sh --dataset_location=/path/to/dataset --input_model=/path/to/model --mode=benchmark
```