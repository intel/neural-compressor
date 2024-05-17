import pytest
import torch
import torch.testing._internal.common_quantization as torch_test_quant_common

from neural_compressor.torch import export
from neural_compressor.torch import utils as torch_utils
from neural_compressor.torch.algorithms.pt2e_quant import half_precision_rewriter


class TestHalfPrecisionConverter(torch_test_quant_common.QuantizationTestCase):

    @staticmethod
    def build_simple_torch_model_and_example_inputs():
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.fc2(x)
                return x

        model = SimpleModel()
        example_inputs = (torch.randn(10, 10),)
        return model, example_inputs

    @pytest.mark.skipif(
        torch_utils.get_torch_version() <= torch_utils.TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0"
    )
    def test_quantizer_on_simple_model(self):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        exported_model = export.export_model_for_pt2e_quant(model=model, example_inputs=example_inputs)
        print("Exported model:")
        exported_model.print_readable()
        unquantized_node_set = half_precision_rewriter.get_unquantized_node_set(exported_model)
        print("Before apply half precision rewriter:")
        exported_model.print_readable(True)
        half_precision_rewriter.transformation(exported_model, unquantized_node_set)
        print("After apply half precision rewriter:")
        exported_model.print_readable(True)
        expected_node_occurrence = {
            # 4 `aten.to` for each `aten.linear`
            torch.ops.aten.to.dtype: 8,
            torch.ops.aten.linear.default: 2,
        }
        expected_node_occurrence = {
            torch_test_quant_common.NodeSpec.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        self.checkGraphModuleNodes(exported_model, expected_node_occurrence=expected_node_occurrence)
