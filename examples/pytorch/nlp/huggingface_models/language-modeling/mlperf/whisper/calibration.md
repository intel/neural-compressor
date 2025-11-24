## Intel MLPerf Inference Calibration and Quantization Details

### RetinaNet Quantization
Model Source: https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/retinanet/pytorch-cpu/scripts/run_calibration.sh

### DLRMv2 Quantization
Model Source: https://zenodo.org/record/5597155

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/dlrm-v2-99.9/pytorch-cpu/scripts/run_calibration.sh

### R-GAT Quantization
Model Source: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/

Model Quantization: FP32 -> INT8

Implementation: /closed/Intel/code/rgat/pytorch-cpu/backend.py

### Whisper Quantization
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT8

Details: /closed/Intel/code/whisper/pytorch-cpu/scripts/run_calibration.sh

### Llama3.1-8B Quantization
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT4

Details: /closed/Intel/code/llama3.1-8b/pytorch-cpu/scripts/run_calibration.sh

