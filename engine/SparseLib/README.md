Just-in-time Deep Neural Network Library (SparseLib)
===========================================

## Abstract

SparseLib is a high-performance operator computing library implemented by assembly. SparseLib contains a JIT domain, a kernel domain, and a scheduling proxy framework.

## Installation
### Build
```
cd SparseLib/
mkdir build
cd build
cmake ..
make -j
```

### Test
After "Build" part,
```
./tests/gtests/test_spmm_sparsednn_prim
```

## API reference for users
### sparse_matmul kernel:
```cpp
#include "interface.hpp"
  ...
  operator_config op_cfg(ker_kind, ker_hypotype, eng_kind, ts_cfgs, op_attrs);
  sparse_matmul_desc spmm_desc(op_cfg);
  sparse_matmul spmm_kern(spmm_desc);

  std::vector<const void*> rt_data = {data0, data1, data2, data3, data4};
  spmm_kern.execute(rt_data);
```
See test_spmm_sparsednn_prim.cpp for details.

## Developer guide for developers
* The jit_domain/ directory, containing different JIT assemblies (Derived class of Xbyak::CodeGenerator).
* The kernels/ directory, containing derived classes of different kernels.
* For different kernels: by convention,
  1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
  2. xxxx_k_t is a specific derived primitive/kernel.
  3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
  where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
* The kernel is determined by the kernel main-kind and kernel isomer. After determining the kernel, it can be implemented by different algorithms. This is the design logic.
