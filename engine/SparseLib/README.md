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
```
cd test/gtest/SparseLib/
mkdir build
cd build
cmake ..
make -j
./test_spmm_default_kernel
```

## API reference for users
### sparse_matmul kernel:
```cpp
#include "interface.hpp"
  ...
  operator_desc op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);
  sparse_matmul_desc spmm_desc(op_desc);
  sparse_matmul spmm_kern(spmm_desc);

  std::vector<const void*> rt_data = {data0, data1, data2, data3, data4};
  spmm_kern.execute(rt_data);
```
See test_spmm_default_kernel.cpp for details.

## Developer guide for developers
* The jit_domain/ directory, containing different JIT assemblies (Derived class of Xbyak::CodeGenerator).
* The kernels/ directory, containing derived classes of different kernels.
* For different kernels: by convention,
  1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
  2. xxxx_k_t is a specific derived primitive/kernel.
  3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
  where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
* The kernel is determined by the kernel main-kind and kernel isomer. After determining the kernel, it can be implemented by different algorithms. This is the design logic.
