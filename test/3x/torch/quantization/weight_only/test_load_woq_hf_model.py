import torch

from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()


class TestHFModelLoad:
    def setup_class(self):
        self.model_name = "TheBloke/TinyLlama-1.1B-python-v0.1-GPTQ"
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)

    def test_load_hf_woq_model(self):
        from neural_compressor.torch.quantization import load

        qmodel = load(model_name_or_path=self.model_name, format="huggingface", torch_dtype=torch.float32)

        woq_linear_num = 0
        for _, module in qmodel.named_modules():
            if module.__class__.__name__ == "WeightOnlyLinear":
                woq_linear_num += 1
        assert woq_linear_num == 154, "Incorrect number of WeightOnlyLinear modules"
        output = qmodel(self.example_inputs)[0]
        assert len(output) > 0, "Not loading the model correctly"
