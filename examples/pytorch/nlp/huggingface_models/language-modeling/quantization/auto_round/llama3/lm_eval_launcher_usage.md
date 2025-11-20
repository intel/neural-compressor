# LM-Eval Launcher Usage

This launcher script (`lm_eval_launcher.py`) helps you run lm_eval with Auto-Round extension support for vLLM models.

## Basic Usage

```bash
# Enable Auto-Round extension and run evaluation
python lm_eval_launcher.py --enable-ar-ext --model vllm \
    --model_args pretrained=OPT-125m-MXFP4,tensor_parallel_size=1,data_parallel_size=1 \
    --tasks lambada_openai \
    --batch_size 8
```

## Alternative: Using environment variable

```bash
# Set environment variable to enable extension
export VLLM_ENABLE_AR_EXT=1

# Run evaluation (extension will be automatically loaded)
python lm_eval_launcher.py --model vllm \
    --model_args pretrained=OPT-125m-MXFP4,tensor_parallel_size=1,data_parallel_size=1 \
    --tasks lambada_openai \
    --batch_size 8
```

## Multiple Tasks

```bash
# Run multiple evaluation tasks
python lm_eval_launcher.py --enable-ar-ext --model vllm \
    --model_args pretrained=your_model_path,tensor_parallel_size=1 \
    --tasks lambada_openai,mmlu,hellaswag \
    --batch_size 8
```

## Available Options

### Launcher-specific options:
- `--enable-ar-ext`: Enable Auto-Round extension for vLLM

### All other options are passed directly to lm_eval:
- `--model`: Model type (e.g., vllm, hf)
- `--model_args`: Model arguments (pretrained path, tensor_parallel_size, etc.)
- `--tasks`: Evaluation tasks to run
- `--batch_size`: Batch size for evaluation
- `--limit`: Limit number of examples per task
- And all other lm_eval options...

## Prerequisites

Make sure you have:
1. lm_eval installed: `pip install lm_eval`
2. Auto-Round extension installed and available
3. vLLM with Auto-Round support installed

## Environment Variables

- `VLLM_ENABLE_AR_EXT=1`: Enable Auto-Round extension
- `CUDA_VISIBLE_DEVICES`: Set GPU devices to use