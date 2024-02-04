import copy
import unittest
import transformers
import torch
from neural_compressor.torch.quantization import TEQConfig, quantize
from neural_compressor.torch.algorithms.weight_only.teq import teq_quantize_impl

class TestTEQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.gptj.seqlen = 512

    def generate_random_corpus(self, nsamples=32):
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
        test_input = torch.ones([1, 512], dtype=torch.long)
        model = copy.deepcopy(self.gptj)

        weight_config = {
            # 'op_name': (bit, group_size, scheme)
            "transformer.h.0.mlp.fc_in": {"bits": 8, "group_size": -1, "scheme": "sym"},
            "transformer.h.0.mlp.fc_out": {"bits": 4, "group_size": 32, "scheme": "asym"},
        }
        absorb_dict = {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]}

        model = teq_quantize_impl(
            model,
            weight_config=weight_config,
            absorb_to_layer=absorb_dict,
            folding=True,
            dataloader=dataloader,
        )
        out1 = model(test_input)
        quant_config = {
            "teq": {
                "global": {
                    "dtype": "fp32",
                },
                "local": {
                    "transformer.h.0.mlp.fc_in": {
                        "dtype": "int",
                        "bits": 8,
                        "group_size": -1,
                        "use_sym": True,
                        "folding": True,
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]},
                        },
                    "transformer.h.0.mlp.fc_out": {
                        "dtype": "int",
                        "bits": 4,
                        "group_size": 32,
                        "use_sym": False,
                        "folding": True,
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]},
                        }
                },
            }
        }
        
        qdq_model = quantize(
                model=self.gptj, quant_config=quant_config, run_args=dataloader
        )
        self.assertTrue(isinstance(qdq_model, torch.nn.Module))
        out2 = qdq_model(test_input)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-02))
        
if __name__ == "__main__":
    unittest.main()