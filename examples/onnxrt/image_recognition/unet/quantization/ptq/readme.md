# Evaluate performance of ONNX Runtime(unet) 

This is an experimental example to quantize unet model. We use dummy data to do quantization and evaluation, so the accuracy is not guaranteed.

### Environment
onnx: 1.12.0
onnxruntime: 1.12.1

### Prepare model

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/scripts/
python convert_stable_diffusion_checkpoint_to_onnx.py --model_path "CompVis/stable-diffusion-v1-4" --output_path /workdir/output_path
```

### Quantization

```bash
bash run_tuning.sh --input_model=/workdir/output_path/unet/model.onnx \ 
                   --config=unet.yaml \ 
                   --output_model=path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=/workdir/output_path/unet/model.onnx \
                      --config=unet.yaml \
                      --mode=performance
```
