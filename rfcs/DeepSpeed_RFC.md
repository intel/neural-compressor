## **Summary**

This is a high level design discussion RFC for contributing some device-agnostic compression algorithms, like the post training static quantization and structural sparsity supported by [Intel(R) Neural Compressor](https://github.com/intel/neural-compressor) into DeepSpeed.

## **Motivation**

As we know, the DeepSpeed Compression have supported many useful compression methods like layer reduction via knowledge distillation, weight quantization, activation quantization, sparse pruning, row pruning, head pruning, and channel pruning.

But those compression methods are lack of main stream support on some popular compression algorithms like post training static quantization and structural sparsity, which have been demostrated as efficient and popular compression methods by the industry.

[Intel(R) Neural Compressor](https://github.com/intel/neural-compressor) has implemented such device-agnostic compression algorithms, we would like to contribute those into DeepSpeed.

## **Proposal Details**

The detail proposal please refer to below RFCs.

[Pruning RFC](./DeepSpeed_Pruning.md): Support structural sparsity capability into DeepSpeed.

[Quantization RFC](./DeepSpeed_Quantization.md): Support post training static quantization into DeepSpeed.

## **Future Works**

The large language model support by post training quantization and structural sparsity is WIP, we will contribute to DeepSpeed when we have some promising results here.