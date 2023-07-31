import sys
sys.path.append("./")
import unittest
import copy
import torch

from neural_compressor.adaptor.torch_utils.weight_only import rtn_quantize, awq_quantize, gptq_quantize, teq_quantize
from neural_compressor.adaptor.torch_utils.smooth_quant import GraphTrace
from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
import transformers


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 32)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class SimpleDataLoader():
    def __init__(self):
        self.batch_size = 1
        self.input = torch.randn([1, 32])

    def __iter__(self):
        yield self.input

class TestAWQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = Model()
        self.dataloader = SimpleDataLoader()
        self.example_inputs = torch.randn([1, 32])
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            'hf-internal-testing/tiny-random-GPTJForCausalLM',
            torchscript=True,
        )
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    def test_trace(self):
        op_types = ['Linear']
        tg = GraphTrace()
        # absorb_to_layer={'absorb_layer': absorbed_layer}
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.model, self.example_inputs, op_types)
        self.assertTrue(len(no_absorb_layers) == 1)
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.gptj, self.lm_input, op_types)
        self.assertTrue(len(no_absorb_layers) == 11)
        return absorb_to_layer, no_absorb_layers

    def test_rtn(self):
        fp32_model = copy.deepcopy(self.model)
        model1 = rtn_quantize(fp32_model, num_bits=3, group_size=-1)
        self.assertTrue(isinstance(model1.fc1, torch.nn.Linear))
        weight_config = {
            # 'op_name': (bit, group_size, sheme)
            'fc1': {
                'bits': 8,
                'group_size': -1,
                'scheme': 'sym'
            },
            'fc2': {
                'bits': 4,
                'group_size': 32,
                'scheme': 'asym',
                'quantile': 0.95, # not required.
            },
        }
        model2 = rtn_quantize(fp32_model, weight_config=weight_config)
        model2 = rtn_quantize(fp32_model, weight_config=weight_config, return_int=True)
        self.assertTrue(isinstance(model2.fc1, WeightOnlyLinear))


    def test_awq(self):
        fp32_model = copy.deepcopy(self.model)
        weight_config = {
            # 'op_name': (bit, group_size, sheme)
            'fc1': {
                'bits': 8,
                'group_size': -1,
                'scheme': 'sym'
            },
            'fc2': {
                'bits': 4,
                'group_size': 32,
                'scheme': 'asym'
            },
        }
        absorb_dict = {
            'fc1': ['fc2']
        }
        model1 = awq_quantize(
            fp32_model, 
            weight_config=weight_config, 
            absorb_dict=absorb_dict, 
            dataloader=self.dataloader, 
            n_samples=128, 
            auto_scale=True, 
            mse_range=True, 
        )
        self.assertTrue(isinstance(model1.fc1, torch.nn.Linear))

        fp32_model = copy.deepcopy(self.model)
        model2 = awq_quantize(
            fp32_model, 
            weight_config=weight_config, 
            absorb_dict=absorb_dict, 
            dataloader=self.dataloader, 
            n_samples=128, 
            auto_scale=True, 
            mse_range=True, 
            return_int=True
        )
        self.assertTrue(isinstance(model2.fc1, WeightOnlyLinear))

class TestGPTQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            'hf-internal-testing/tiny-random-GPTJForCausalLM',
            torchscript=True,
        )
        self.gptj.seqlen = 512
    
    def generate_random_corpus(self, nsamples = 32):
        meta_data = []
        for _ in range(nsamples):
            inp = torch.ones([1, 512], dtype=torch.long)
            tar = torch.ones([1, 512], dtype=torch.long)
            meta_data.append((inp, tar))
        return meta_data

    def test_gptq(self):
        dataloader = self.generate_random_corpus()
        model = copy.deepcopy(self.gptj)
        weight_config = {
            'transformer.h.0.attn.k_proj':{
                'wbits': 4,
                'group_size': 128,
                'sym': True,
                'percdamp': 0.01,
                'perchannel': False
            },
            'transformer.h.1.attn.k_proj':{
                'wbits': 3,
                'group_size': -1,
                'sym': False,
                'percdamp': 0.01,
                'actorder': True,
            },
            'transformer.h.1.attn.k_proj':{
                'wbits': 3,
                'group_size': 32,
                'sym': False,
                'percdamp': 0.01,
                'mse': True,
                'actorder': False
            },
        }
        quantizer = gptq_quantize(model, weight_config=weight_config, dataloader=dataloader, )
        self.assertTrue(isinstance(model, torch.nn.Module))
        del model

        model = copy.deepcopy(self.gptj)
        weight_config = {
            "wbits": 4
        }
        quantizer = gptq_quantize(model, weight_config=weight_config, dataloader=dataloader, )
        self.assertTrue(isinstance(model, torch.nn.Module))
        del model

class TestTEQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            'hf-internal-testing/tiny-random-GPTJForCausalLM',
            torchscript=True,
        )
        self.gptj.seqlen = 512

    def generate_random_corpus(self, nsamples = 32):
        meta_data = []
        for _ in range(nsamples):
            inp = torch.ones([1, 512], dtype=torch.long)
            tar = torch.ones([1, 512], dtype=torch.long)
            meta_data.append((inp, tar))
        return meta_data

    def train_func(self):
        pass

    def test_teq(self):
        dataloader = self.generate_random_corpus()
        model = copy.deepcopy(self.gptj)

        weight_config = {
            # 'op_name': (bit, group_size, sheme)
            'transformer.h.0.mlp.fc_in': {
                'bits': 8,
                'group_size': -1,
                'scheme': 'sym'
            },
            'transformer.h.0.mlp.fc_out': {
                'bits': 4,
                'group_size': 32,
                'scheme': 'asym'
            },
        }
        absorb_dict = {
            'transformer.h.0.mlp.fc_in': ['transformer.h.0.mlp.fc_out']
        }
        extra_config = {'folding': True}


        model = teq_quantize(model, weight_config=weight_config, absorb_to_layer=absorb_dict,
                extra_config=extra_config, dataloader=dataloader)
        self.assertTrue(isinstance(model, torch.nn.Module))

if __name__ == "__main__":
    unittest.main()
