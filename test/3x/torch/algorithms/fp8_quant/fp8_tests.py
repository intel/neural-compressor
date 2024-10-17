import habana_frameworks.torch.core as htcore
import torch

import neural_compressor.torch.algorithms.fp8_quant

# This file is for small tests run for debug flow and accuracy. (Not for CI)


class TinyBlock(torch.nn.Module):

    def __init__(self):
        super(TinyBlock, self).__init__()
        self.pre_linear = torch.nn.Linear(2, 1, bias=False)
        self.pre_linear.weight = torch.nn.Parameter(torch.ones([1, 2]))

    def forward(self, x):
        x = self.pre_linear(x)
        return x


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        self.block = TinyBlock()

    def forward(self, x):
        x = self.block(x)
        return x


class TinyBlock2(torch.nn.Module):

    def __init__(self):
        super(TinyBlock2, self).__init__()
        self.pre_linear = torch.nn.Linear(2, 1, bias=False)
        self.pre_linear.weight = torch.nn.Parameter(torch.ones([1, 2]))
        self.pre_linear2 = torch.nn.Linear(1, 1, bias=False)
        self.pre_linear2.weight = torch.nn.Parameter(torch.ones([1, 1]))

    def forward(self, x):
        x = self.pre_linear(x)
        x = self.pre_linear2(x)
        return x


class TinyModel2(torch.nn.Module):

    def __init__(self):
        super(TinyModel2, self).__init__()
        self.block = TinyBlock2()

    def forward(self, x):
        x = self.block(x)
        return x


class TinyModel3(torch.nn.Module):

    def __init__(self):
        super(TinyModel3, self).__init__()
        self.block = TinyBlock()
        self.block2 = TinyBlock2()

    def forward(self, x, b):
        if b:
            x = self.block(x)
        else:
            x = self.block2(x)
        return x


model = TinyModel()
model.eval()
model = model.to("hpu").to(torch.bfloat16)
htcore.hpu_initialize()
neural_compressor.torch.algorithms.fp8_quant.prep_model(model)  # fp8 additions


with torch.no_grad():

    # >>> new_fp8converted_input = (torch.tensor(MaxAbs(input), dtype=torch.bfloat16) / torch.tensor(InputScale, dtype=torch.bfloat16)).to(torch.float8_e4m3fn)
    # >>> new_fp8converted_weight = (torch.tensor(MaxAbs(weight), dtype=torch.bfloat16) / torch.tensor(WeightScale, dtype=torch.bfloat16)).to(torch.float8_e4m3fn)
    # >>> mul_result = new_fp8converted_weight.to(torch.bfloat16) * new_fp8converted_input.to(torch.bfloat16)
    # >>> result = mul_result * torch.tensor(InputScale, dtype=torch.bfloat16) * torch.tensor(WeightScale, dtype=torch.bfloat16)

    # If the results of the first 2 lines > 240 (or nan), assume they are equal to 240. (In G2 or G3 with specific fp8 representation settings)

    # Run simulator:
    # Gaudi2: run_coral_sim --chip-type gaudi2 -r -D 32
    # Gaudi3: run_coral_sim --chip-type gaudi3 -r -D 32
    # cd .../quantization_toolkit/habana_quantization_toolkit/tests/

    # Test1: (Disable (comment) all other tests, delete all files from the test_outputs folder)
    # Run:
    # QUANT_CONFIG=test_jsons/test_measure.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_hw_quant.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_pow2_quant.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_unit_quant.json python3 fp8_tests.py

    out_arange = model((torch.tensor([[232, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    out_arange = model((torch.tensor([[240, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    out_arange = model((torch.tensor([[248, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    # Result (Same for Gaudi2 and Gaudi3):
    # for HW/POW2:
    # tensor([[224.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[256.]], device='hpu:0', dtype=torch.bfloat16)
    # for Unit:
    # tensor([[224.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)

    # Test2: (Disable (comment) all other tests, delete all files from the test_outputs folder)
    # Run:
    # QUANT_CONFIG=test_jsons/test_measure.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_hw_quant.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_pow2_quant.json python3 fp8_tests.py
    # QUANT_CONFIG=test_jsons/test_unit_quant.json python3 fp8_tests.py

    out_arange = model((torch.tensor([[3720, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    out_arange = model((torch.tensor([[3721, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    out_arange = model((torch.tensor([[13721, 0]], dtype=torch.bfloat16)).to("hpu"))
    print(out_arange)

    # Result:
    # for HW (Gaudi2):
    # tensor([[3584.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[3840.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[3840.]], device='hpu:0', dtype=torch.bfloat16)
    # for HW (Gaudi3) and Pow2:
    # tensor([[3584.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[3840.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[13312.]], device='hpu:0', dtype=torch.bfloat16)
    # for Unit:
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)
    # tensor([[240.]], device='hpu:0', dtype=torch.bfloat16)

    # Test3: (Disable (comment) all other tests, delete all files from the test_outputs folder)
    #        (Change Line 73 above to: model = TinyModel3())
    # Run: (add LOG_LEVEL_INC=0/1 for additional logs)
    #      (Uncomment lines 164+165)
    # 1) QUANT_CONFIG=test_jsons/test_measure.json python3 fp8_tests.py
    # 2) QUANT_CONFIG=test_jsons/test_hw_quant.json python3 fp8_tests.py
    # 3) QUANT_CONFIG=test_jsons/test_hw_quant_ignored_unmeasured_models.json python3 fp8_tests.py
    # (Comment lines 164+165, Uncomment lines 166+167)
    # 4) QUANT_CONFIG=test_jsons/test_hw_quant.json python3 fp8_tests.py
    # 5) QUANT_CONFIG=test_jsons/test_hw_quant_ignored_unmeasured_models.json python3 fp8_tests.py

    # out_arange = model((torch.tensor([[232, 0]], dtype=torch.bfloat16)).to('hpu'), True)
    # print(out_arange)
    # out_arange = model((torch.tensor([[232, 0]], dtype=torch.bfloat16)).to('hpu'), False)
    # print(out_arange)

    # Result:
    # 1) tensor([[232.]], device='hpu:0', dtype=torch.bfloat16)
    # 2) tensor([[224.]], device='hpu:0', dtype=torch.bfloat16)
    # 3) tensor([[224.]], device='hpu:0', dtype=torch.bfloat16)
    # 4) Exception: Error - Layer 'block2.pre_linear' was called but was not quantized because no measures were supplied.
    # 5) tensor([[232.]], device='hpu:0', dtype=torch.bfloat16)

    # fp8 additions
    neural_compressor.torch.algorithms.fp8_quant.finish_measurements(model)
