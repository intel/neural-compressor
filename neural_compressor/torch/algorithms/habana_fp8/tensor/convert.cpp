//  Copyright (c) 2024 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Temporary implementation of fp8 tensor saving and loading
// Will remove after Habana torch applies below patch:
// https://github.com/pytorch/pytorch/pull/114662


#include <torch/extension.h>


// function prototype declaration
torch::Tensor to_u8(torch::Tensor tensor);
torch::Tensor from_u8(torch::Tensor tensor, int choice=1);


torch::Tensor to_u8(torch::Tensor tensor) {
    auto p = tensor.data_ptr();
    // RuntimeError: HPU device type not enabled.
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8);
    auto tmp = torch::from_blob(p, tensor.sizes(), options);
    // copy to avoid memory leak.
    torch::Tensor tensor_uint8 = torch::empty_like(tensor, torch::kUInt8).copy_(tmp);
    return tensor_uint8;
};


/*
choice=1 means torch.float8_e4m3fn;
others means torch.float8_e5m2;
*/
torch::Tensor from_u8(torch::Tensor tensor, int choice) {
    auto p = tensor.data_ptr();
    torch::ScalarType dtype;
    if (choice == 1) {
        dtype = torch::kFloat8_e4m3fn;
    }
    else {
        dtype = torch::kFloat8_e5m2;
    }
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(dtype);
    auto tmp = torch::from_blob(p, tensor.sizes(), options);
    // copy to avoid memory leak.
    torch::Tensor tensor_fp8 = torch::empty_like(tensor, dtype).copy_(tmp);
    return tensor_fp8;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_u8", &to_u8, "Convert tensor to u8 for saving.");
    m.def("from_u8", &from_u8, "Recover tensor from u8 for loading.");
};
