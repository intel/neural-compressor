import copy
import shutil

import huggingface_hub
import pytest
import torch
import transformers

from neural_compressor.torch.quantization import load
from neural_compressor.torch.utils import SaveLoadFormat, accelerator, is_hpu_available

device = accelerator.current_device_name()


class TestHFModelLoad:
    def setup_class(self):
        self.model_name = "TheBloke/TinyLlama-1.1B-python-v0.1-GPTQ"
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
        self.local_cache = "local_cache"

        self.local_hf_model = "TinyLlama-1.1B-Chat-v0.1-GPTQ"
        huggingface_hub.snapshot_download(self.model_name, local_dir=self.local_hf_model)

    def teardown_class(self):
        shutil.rmtree("TinyLlama-1.1B-python-v0.1-GPTQ", ignore_errors=True)
        shutil.rmtree("saved_results", ignore_errors=True)
        shutil.rmtree("nc_workspace", ignore_errors=True)
        shutil.rmtree("local_cache", ignore_errors=True)

    def get_woq_linear_num(self, model, woq_module_type_name):
        woq_linear_num = 0
        for _, module in model.named_modules():
            if module.__class__.__name__ == woq_module_type_name:
                woq_linear_num += 1
        return woq_linear_num

    def test_load_hf_woq_model_cpu(self):
        # use huggingface model_id (format=huggingface, device="cpu")
        qmodel = load(
            model_name_or_path=self.model_name, format="huggingface", torch_dtype=torch.float32
        )  # 'torch_dtype=torch.float32' for cpu test
        assert (
            self.get_woq_linear_num(qmodel, "INCWeightOnlyLinear") == 154
        ), "Incorrect number of INCWeightOnlyLinear modules"
        output = qmodel(self.example_inputs.to("cpu"))[0]
        assert len(output) > 0, "Not loading the model correctly"

        # use huggingface local model_path (format=huggingface, device="cpu")
        qmodel = load(
            model_name_or_path=self.local_hf_model, format="huggingface", torch_dtype=torch.float32
        )  # 'torch_dtype=torch.float32' for cpu test
        assert (
            self.get_woq_linear_num(qmodel, "INCWeightOnlyLinear") == 154
        ), "Incorrect number of INCWeightOnlyLinear modules"
        output = qmodel(self.example_inputs.to("cpu"))[0]
        assert len(output) > 0, "Not loading the model correctly"

    @pytest.mark.skipif(not is_hpu_available(), reason="no hpex in environment here.")
    def test_load_hf_woq_model_hpu(self):
        # use huggingface model_id (format=huggingface, device="hpu")
        # first load: linear -> INCWeightOnlyLinear -> HPUWeightOnlyLinear, save hpu_model.safetensors to local cache dir
        model = load(
            model_name_or_path=self.model_name,
            format="huggingface",
            device="hpu",
            torch_dtype=torch.bfloat16,
            cache_dir=self.local_cache,
        )
        assert (
            self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154
        ), "Incorrect number of HPUWeightOnlyLinear modules"
        output1 = model(self.example_inputs)[0]

        # second load: linear -> HPUWeightOnlyLinear using hpu_model.safetensors saved in local cache dir
        model = load(
            model_name_or_path=self.model_name,
            format="huggingface",
            device="hpu",
            torch_dtype=torch.bfloat16,
            cache_dir=self.local_cache,
        )
        assert (
            self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154
        ), "Incorrect number of HPUWeightOnlyLinear modules"
        output2 = model(self.example_inputs)[0]

        assert torch.equal(
            output1, output2
        ), "The model loaded the second time is different from the model loaded the first time"

    @pytest.mark.skipif(not is_hpu_available(), reason="no hpex in environment here.")
    def test_load_hf_woq_model_hpu_special_case(self):
        # this model contains tensors sharing memory
        model = load(
            model_name_or_path="ybelkada/opt-125m-gptq-4bit",
            format="huggingface",
            device="hpu",
            torch_dtype=torch.bfloat16,
            cache_dir=self.local_cache,
        )
        assert model is not None, "Model not loaded correctly"
