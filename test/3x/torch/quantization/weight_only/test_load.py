import torch
import shutil
import copy
import transformers
import huggingface_hub

from neural_compressor.common import logger
from neural_compressor.torch.utils import LoadFormat
from neural_compressor.torch.algorithms.weight_only.save_load import WOQModelLoader
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()

class TestHFModelLoad:
    def setup_class(self):
        self.model_name = "TheBloke/TinyLlama-1.1B-python-v0.1-GPTQ"
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)

        self.local_hf_model = "./TinyLlama-1.1B-python-v0.1-GPTQ"
        huggingface_hub.snapshot_download(self.model_name, local_dir=self.local_hf_model)

        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )

        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
            device_map=device,
        )
        self.gptj.seqlen = 512

    def teardown_class(self):
        shutil.rmtree("TinyLlama-1.1B-python-v0.1-GPTQ", ignore_errors=True)
        shutil.rmtree("saved_results", ignore_errors=True)

    def get_woq_linear_num(self, model, woq_module_type_name):
        woq_linear_num = 0
        for _, module in model.named_modules():
            if module.__class__.__name__ == woq_module_type_name:
                woq_linear_num += 1
        return woq_linear_num

    def test_load_hf_woq_model(self):
        from neural_compressor.torch.quantization import load

        # 1. huggingface model_id (format=huggingface, device="cpu")
        qmodel = load(model_name_or_path=self.model_name, format="huggingface", torch_dtype=torch.float32) # 'torch_dtype=torch.float32' for cpu test
        assert self.get_woq_linear_num(qmodel, "INCWeightOnlyLinear") == 154, "Incorrect number of INCWeightOnlyLinear modules"
        output = qmodel(self.example_inputs)[0]
        assert len(output) > 0, "Not loading the model correctly"

        # 2. huggingface model_id (format=huggingface, device="hpu")
        # first load: linear -> INCWeightOnlyLinear -> HPUWeightOnlyLinear, save hpu_model.safetensors to local cache dir
        model_loader = WOQModelLoader(model_name_or_path=self.model_name, format=LoadFormat.HUGGINGFACE, device="hpu", torch_dtype=torch.float32)
        model = model_loader.load_woq_model()
        assert self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154, "Incorrect number of HPUWeightOnlyLinear modules"
        output1 = model(self.example_inputs)[0]

        # second load: linear -> HPUWeightOnlyLinear using hpu_model.safetensors saved in local cache dir
        model_loader = WOQModelLoader(model_name_or_path=self.model_name, format=LoadFormat.HUGGINGFACE, device="hpu", torch_dtype=torch.float32)
        model = model_loader.load_woq_model()
        assert self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154, "Incorrect number of HPUWeightOnlyLinear modules"
        output2 = model(self.example_inputs)[0]

        assert torch.equal(output1, output2), "The model loaded the second time is different from the model loaded the first time"

        # 3. huggingface local model_path (format=huggingface, device="hpu")
        # first load: linear -> INCWeightOnlyLinear -> HPUWeightOnlyLinear, save hpu_model.safetensors to local cache dir
        model_loader = WOQModelLoader(model_name_or_path=self.local_hf_model, format=LoadFormat.HUGGINGFACE, device="hpu", torch_dtype=torch.float32)
        model = model_loader.load_woq_model()
        assert self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154, "Incorrect number of HPUWeightOnlyLinear modules"
        output1 = model(self.example_inputs)[0]

        # second load: linear -> HPUWeightOnlyLinear using hpu_model.safetensors saved in local cache dir
        model_loader = WOQModelLoader(model_name_or_path=self.local_hf_model, format=LoadFormat.HUGGINGFACE, device="hpu", torch_dtype=torch.float32)
        model = model_loader.load_woq_model()
        assert self.get_woq_linear_num(model, "HPUWeightOnlyLinear") == 154, "Incorrect number of HPUWeightOnlyLinear modules"
        output2 = model(self.example_inputs)[0]

        assert torch.equal(output1, output2), "The model loaded the second time is different from the model loaded the first time"


test = TestHFModelLoad()
test.setup_class()
test.test_load_hf_woq_model()
test.teardown_class()
