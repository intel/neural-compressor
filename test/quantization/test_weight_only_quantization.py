import sys
sys.path.append("./")
import unittest
import copy
import torch
from neural_compressor.adaptor.torch_utils.weight_only import rtn_quantize, gptq_quantize
import transformers

class TestWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 2, 2)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(4, 10, 3, 3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out) + x
                return out

        self.model = Model()

    @classmethod
    def tearDownClass(self):
        pass

    def test_conv(self):
        fp32_model = copy.deepcopy(self.model)
        model1 = rtn_quantize(fp32_model, num_bits=3, group_size=-1)
        w_layers_config = {
            # 'op_name': (bit, group_size, sheme)
            'conv1': (8, 128, 'sym'),
            'conv2': (4, 32, 'asym')
        }
        model2 = rtn_quantize(fp32_model, num_bits=3, group_size=-1, w_layers_config=w_layers_config)


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
        fp32_model = copy.deepcopy(self.gptj)
        weight_config = {
            'wbits': 4,
            'group_size': 128,
            'sym': False,
            'percdamp': 0.01,
        }
        # import pdb;pdb.set_trace()
        model = gptq_quantize(fp32_model, weight_config=weight_config, dataloader=dataloader, )

if __name__ == "__main__":
    unittest.main()